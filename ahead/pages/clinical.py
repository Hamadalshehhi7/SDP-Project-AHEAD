"""
ahead/pages/clinical.py
=======================
Doctor / Admin only: upload a disease-specific patient file, screen the whole
file or one record, and review the results.
"""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ahead.components import kpi_card, model_selector, page_header, result_card, section_heading
from ahead.config import DISEASES, GENDER_CODED_FIELDS, pretty_label
from ahead.resources import feature_summary, features, predict_probability

ID_COLUMNS = ["PatientID", "patient_id", "Patient ID", "ID", "id"]


def read_patient_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


MAX_ROWS = 5000


def _clean(value) -> str:
    """Make an arbitrary uploaded value safe to echo inside a Markdown warning."""
    return str(value).replace("`", "'").replace("\n", " ")[:60]


def _validate(frame: pd.DataFrame, disease: str, required: list):
    """
    Coerce numeric columns and report values the model cannot use: unparseable
    numbers (imputed by the pipeline) and categories never seen in training
    (ignored by the one-hot encoder, i.e. treated as "none of the known values").
    """
    frame = frame.copy()
    problems = []
    stats = feature_summary(disease)
    for column in required:
        info = stats.get(column)
        if info is None:
            continue
        if info["kind"] == "number":
            if info.get("binary"):
                # accept Yes/No, True/False and Male/Female spellings in 0/1-coded columns
                tokens = {"yes": 1, "true": 1, "y": 1, "no": 0, "false": 0, "n": 0}
                if column in GENDER_CODED_FIELDS:
                    tokens.update({"male": 0, "m": 0, "female": 1, "f": 1})
                frame[column] = frame[column].map(
                    lambda v: tokens.get(str(v).strip().lower(), v) if isinstance(v, str) else v
                )
            coerced = pd.to_numeric(frame[column], errors="coerce")
            bad = int(coerced.isna().sum() - frame[column].isna().sum())
            if bad:
                problems.append(f"**{pretty_label(column)}**: {bad} value(s) are not numeric and will be treated as missing.")
            frame[column] = coerced
        else:
            known = {option.strip().lower(): option for option in info["options"]}
            # normalise case / surrounding spaces to the training spelling
            frame[column] = frame[column].map(
                lambda v: known.get(str(v).strip().lower(), v) if pd.notna(v) else v
            )
            values = frame[column].dropna().astype(str)
            unknown = sorted(set(values) - set(known.values()))
            if unknown:
                shown = ", ".join(f"`{_clean(u)}`" for u in unknown[:5]) + (" …" if len(unknown) > 5 else "")
                problems.append(
                    f"**{pretty_label(column)}**: unknown value(s) {shown}. Expected one of: "
                    + ", ".join(f"`{o}`" for o in sorted(known.values())) + "."
                )
    missing = int(frame[required].isna().sum().sum())
    if missing:
        problems.append(f"{missing} missing value(s) across the required columns will be imputed with training-set defaults.")
    return frame, problems


def render_clinical_dashboard() -> None:
    page_header(
        "Hospital Clinical Decision Support", "Clinical Dashboard",
        "Upload a disease-specific patient file, screen every record or a single patient, and review "
        "AHEAD model output for clinical follow-up.",
    )
    st.warning(
        "AHEAD is a research and educational decision-support prototype. Its output is not a confirmed "
        "diagnosis and must be interpreted by a qualified clinician."
    )

    c1, c2 = st.columns(2)
    with c1:
        disease = st.selectbox(
            "Disease screening", list(DISEASES), format_func=lambda d: DISEASES[d]["card_title"], key="clinical_disease",
        )
    with c2:
        pipeline, model_name, threshold = model_selector(disease, key=f"clinical_model_{disease}")
    if pipeline is None:
        st.error("No trained model is available. Run `python train_model.py` first.")
        return

    required = features(disease)
    uploaded = st.file_uploader(
        "Upload patient records", type=["csv", "xlsx"],
        help="One disease-specific file. Each row represents one patient record.",
    )
    with st.expander("Required columns for this screening"):
        st.code(", ".join(required), language=None)

    if uploaded is None:
        st.info("Upload a CSV or Excel file to begin.")
        return

    try:
        patient_df = read_patient_file(uploaded)
    except Exception as error:
        st.error("AHEAD could not read this file. Please upload a valid CSV or XLSX export.")
        st.caption(f"Details: {escape(str(error))[:300]}")
        return

    missing = [column for column in required if column not in patient_df.columns]
    if missing:
        st.error("The uploaded file is missing required columns: " + ", ".join(missing))
        return
    if patient_df.empty:
        st.error("The uploaded file has no patient rows.")
        return
    if len(patient_df) > MAX_ROWS:
        st.warning(f"Only the first {MAX_ROWS:,} of {len(patient_df):,} rows are screened in this prototype.")
        patient_df = patient_df.head(MAX_ROWS)

    patient_df, problems = _validate(patient_df, disease, required)
    for problem in problems:
        st.warning(problem)

    id_column = next((c for c in ID_COLUMNS if c in patient_df.columns), None)
    st.success(f"File loaded — {len(patient_df):,} patient records found.")

    # ------------------------------------------------------------------ batch screening
    section_heading("Cohort screening", f"Every record scored with {model_name} at a {threshold * 100:.0f}% threshold.")
    # Score once per (file, disease, model); reruns caused by widget clicks reuse the result.
    cache = st.session_state.setdefault("cohort_scores", {})
    cache_key = (uploaded.file_id, disease, model_name)
    if cache_key not in cache:
        cache.clear()
        cache[cache_key] = predict_probability(pipeline, patient_df[required])
    scores = cache[cache_key]
    flagged = (scores >= threshold)
    row_labels = pd.Series([f"Row {i + 1}" for i in range(len(patient_df))], index=patient_df.index)
    patient_ids = patient_df[id_column].astype(str) if id_column else None
    results = pd.DataFrame({"Record": row_labels})
    if patient_ids is not None:
        results["Patient"] = patient_ids
    results["Model score"] = (scores * 100).round(1)
    results["Flagged"] = flagged.map({True: "Flagged for review", False: "Not flagged"})

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Records screened", f"{len(patient_df):,}", uploaded.name, "▤")
    with k2:
        kpi_card("Flagged for review", f"{int(flagged.sum()):,}", f"{100 * flagged.mean():.1f}% of records", "!", "danger")
    with k3:
        kpi_card("Median score", f"{scores.median() * 100:.1f}%", "Across the uploaded cohort", "◔")

    st.dataframe(
        results.sort_values("Model score", ascending=False),
        width="stretch", hide_index=True, height=300,
        column_config={"Model score": st.column_config.ProgressColumn("Model score", format="%.1f%%", min_value=0, max_value=100)},
    )
    export = results.assign(Model=model_name, Threshold=f"{threshold:.2f}", Condition=DISEASES[disease]["label"])
    st.download_button(
        "Download screening results (CSV)",
        export.to_csv(index=False).encode(),
        file_name=f"ahead_{disease}_screening.csv",
        mime="text/csv",
    )

    # ------------------------------------------------------------------ single record
    section_heading("Single-record review", "Inspect one patient in detail.")
    labels = (row_labels + (" — " + patient_ids if patient_ids is not None else "")).tolist()
    selected_label = st.selectbox("Select patient record", labels, key="clinical_record")
    row_index = labels.index(selected_label)
    selected = patient_df.iloc[[row_index]][required].copy()

    probability = float(scores.iloc[row_index])
    prediction = int(probability >= threshold)
    left, right = st.columns([0.55, 0.45])
    with left:
        result_card(prediction, probability, DISEASES[disease]["long_label"], model_name, threshold)
    with right:
        summary = selected.T.rename(columns={selected.index[0]: "Value"})
        summary.index = [pretty_label(x) for x in summary.index]
        summary["Value"] = summary["Value"].astype(str)  # mixed types → Arrow-safe
        st.dataframe(summary, width="stretch", height=250)
    st.caption(
        "The model score is an algorithmic screening output, not the patient's true probability of disease "
        "and not a confirmed diagnosis."
    )
