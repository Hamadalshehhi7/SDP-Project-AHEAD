"""
AHEAD
Advanced Health Early Awareness and Disease Detection System

Clinical Screening Portal
Senior Design Project
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai


client = None  # Gemini is initialized only when an AI feature is used.


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AHEAD | Clinical Screening",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

DATASET_FILES = {
    "diabetes": "diabetes.csv",
    "heart": "heart.csv",
    "kidney": "kidney_disease.csv",
}

TARGET_COLUMNS = {
    "diabetes": "diabetes",
    "heart": "HadHeartAttack",
    "kidney": "Diagnosis",
}


# =============================================================================
# STYLE
# =============================================================================

st.html(
"""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top right,
            rgba(24, 125, 131, 0.045),
            transparent 30%
        ),
        #F6F8FB;
}

html, body, [class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    color: #1F2937;
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* --------------------------------------------------------------------------
   SIDEBAR
-------------------------------------------------------------------------- */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0B2942 0%,
            #103B54 62%,
            #0C5058 100%
        );
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10);
}

.sidebar-brand {
    padding: 12px 4px 15px;
}

.brand-symbol {
    width: 46px;
    height: 46px;
    border-radius: 13px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.16);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 15px;
}

.sidebar-brand h2 {
    font-size: 1.55rem;
    margin: 0;
    font-weight: 760;
}

.sidebar-brand p {
    margin: 6px 0 0;
    color: #B8CBD8;
    font-size: 0.76rem;
    line-height: 1.5;
}


/* --------------------------------------------------------------------------
   HERO
-------------------------------------------------------------------------- */

.home-hero {
    background:
        linear-gradient(
            125deg,
            #0D304B,
            #14556B 55%,
            #187D7D
        );
    border-radius: 22px;
    padding: 40px;
    color: white;
    box-shadow:
        0 16px 40px rgba(11,41,66,0.16);
    position: relative;
    overflow: hidden;
    margin-bottom: 26px;
}

.home-hero::before {
    content: "";
    position: absolute;
    width: 330px;
    height: 330px;
    border-radius: 50%;
    border: 65px solid rgba(255,255,255,0.035);
    right: -110px;
    top: -150px;
}

.home-eyebrow,
.header-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.68rem;
    color: #BDE5E2;
    font-weight: 740;
}

.home-hero h1 {
    color: white;
    font-size: 2.55rem;
    margin: 8px 0;
    font-weight: 770;
    letter-spacing: -0.05em;
}

.home-hero p {
    max-width: 760px;
    color: #D9E9EE;
    font-size: 0.96rem;
    line-height: 1.65;
    margin: 0;
}


/* --------------------------------------------------------------------------
   PAGE HEADER
-------------------------------------------------------------------------- */

.clinical-header {
    background: white;
    border: 1px solid #E1E8EF;
    border-radius: 20px;
    padding: 31px 34px;
    margin-bottom: 23px;
    box-shadow:
        0 10px 35px rgba(15,42,67,0.05);
}

.clinical-header .header-eyebrow {
    color: #167B81;
}

.clinical-header h1 {
    margin: 8px 0 0;
    color: #102A43;
    font-size: 2rem;
    font-weight: 760;
    letter-spacing: -0.04em;
}

.clinical-header p {
    max-width: 800px;
    color: #667085;
    font-size: 0.93rem;
    line-height: 1.6;
    margin: 10px 0 0;
}


/* --------------------------------------------------------------------------
   CARDS
-------------------------------------------------------------------------- */

.metric-card,
.condition-card,
.action-card {
    background: white;
    border: 1px solid #E3E9EF;
    border-radius: 16px;
    box-shadow:
        0 6px 24px rgba(15,42,67,0.04);
}

.metric-card {
    padding: 20px 21px;
    min-height: 112px;
}

.metric-label {
    color: #667085;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.69rem;
    font-weight: 650;
}

.metric-value {
    color: #102A43;
    font-size: 1.65rem;
    font-weight: 740;
    margin-top: 9px;
}

.metric-foot {
    color: #168188;
    font-size: 0.67rem;
    margin-top: 4px;
}

.condition-card {
    padding: 24px;
    min-height: 235px;
}

.condition-icon {
    width: 43px;
    height: 43px;
    border-radius: 12px;
    background: #ECF7F7;
    color: #167A7F;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 760;
    margin-bottom: 15px;
}

.condition-card h3 {
    color: #17324D;
    font-size: 1.04rem;
    margin: 0;
}

.condition-card p {
    color: #667085;
    font-size: 0.79rem;
    line-height: 1.65;
    margin-top: 10px;
}

.service-note {
    display: inline-block;
    background: #F1F8F8;
    color: #176B71;
    border: 1px solid #D9ECEE;
    border-radius: 100px;
    padding: 5px 9px;
    font-size: 0.66rem;
    font-weight: 650;
    margin-top: 6px;
}


/* --------------------------------------------------------------------------
   SECTION TITLES
-------------------------------------------------------------------------- */

.section-heading {
    margin: 23px 0 15px;
}

.section-heading h2 {
    color: #17324D;
    font-size: 1.22rem;
    margin: 0;
    font-weight: 720;
}

.section-heading p {
    color: #7C8998;
    margin: 4px 0 0;
    font-size: 0.77rem;
}


/* --------------------------------------------------------------------------
   FORMS
-------------------------------------------------------------------------- */

[data-testid="stForm"] {
    background: white;
    border: 1px solid #E3E9EF;
    border-radius: 18px;
    padding: 12px 23px 24px;
    box-shadow:
        0 6px 25px rgba(15,42,67,0.035);
}

.form-section-title {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 18px 0 17px;
    padding-bottom: 11px;
    border-bottom: 1px solid #E8EDF2;
}

.form-section-number {
    width: 28px;
    height: 28px;
    min-width: 28px;
    border-radius: 8px;
    background: #E8F5F5;
    color: #15747A;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.71rem;
    font-weight: 760;
}

.form-section-title h3 {
    margin: 0;
    color: #17324D;
    font-size: 0.99rem;
}

.form-section-title p {
    margin: 3px 0 0;
    color: #98A2B3;
    font-size: 0.72rem;
}

[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: #344054 !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background: #FBFCFD !important;
    border-radius: 9px !important;
}


/* --------------------------------------------------------------------------
   BUTTON
-------------------------------------------------------------------------- */

.stButton > button,
.stFormSubmitButton > button {
    background:
        linear-gradient(
            135deg,
            #126E75,
            #17848A
        ) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: 650 !important;
}


/* --------------------------------------------------------------------------
   RESULTS
-------------------------------------------------------------------------- */

.result-card {
    background: white;
    border: 1px solid #E3E9EF;
    border-radius: 17px;
    padding: 26px;
    min-height: 250px;
    box-shadow:
        0 7px 28px rgba(15,42,67,0.045);
}

.result-elevated {
    border-top: 4px solid #C75D5D;
}

.result-lower {
    border-top: 4px solid #2E9477;
}

.result-kicker {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7C8798;
    font-size: 0.65rem;
    font-weight: 750;
}

.result-high-title {
    color: #A63E3E;
    font-size: 1.45rem;
    font-weight: 740;
    margin: 8px 0 4px;
}

.result-low-title {
    color: #19775E;
    font-size: 1.45rem;
    font-weight: 740;
    margin: 8px 0 4px;
}

.result-score {
    color: #344054;
    font-size: 0.88rem;
    margin-top: 13px;
}

.result-text {
    color: #667085;
    font-size: 0.78rem;
    line-height: 1.6;
    margin-top: 10px;
}


/* --------------------------------------------------------------------------
   NEXT STEPS
-------------------------------------------------------------------------- */

.action-card {
    padding: 19px 20px;
    margin-bottom: 11px;
}

.action-number {
    display: inline-flex;
    width: 29px;
    height: 29px;
    border-radius: 8px;
    background: #E9F5F5;
    color: #14747A;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 760;
    margin-right: 9px;
}

.action-title {
    color: #17324D;
    font-size: 0.91rem;
    font-weight: 700;
}

.action-text {
    color: #667085;
    font-size: 0.78rem;
    line-height: 1.58;
    margin-top: 8px;
}

.risk-factor-box {
    background: #F8FBFB;
    border: 1px solid #E0EEEE;
    border-radius: 14px;
    padding: 17px 19px;
}


/* --------------------------------------------------------------------------
   URGENT CARE
-------------------------------------------------------------------------- */

.urgent-box {
    background: #FFF5F5;
    border: 1px solid #F1C9C9;
    border-left: 4px solid #C85858;
    border-radius: 11px;
    padding: 15px 17px;
    color: #7A3030;
    font-size: 0.76rem;
    line-height: 1.55;
    margin-top: 18px;
}


/* --------------------------------------------------------------------------
   DISCLAIMER
-------------------------------------------------------------------------- */

.disclaimer {
    background: #FFFAEB;
    border: 1px solid #F3E5B2;
    border-left: 4px solid #D5A43A;
    border-radius: 11px;
    padding: 14px 16px;
    color: #765A1C;
    font-size: 0.74rem;
    line-height: 1.55;
    margin-top: 20px;
}

</style>
"""
)


# =============================================================================
# FEATURE GROUPS
# =============================================================================

FEATURE_GROUPS = {

    "diabetes": {

        "Lifestyle & Patient Profile": [
            "gender",
            "age",
            "smoking_history",
            "bmi",
        ],

        "Medical History": [
            "hypertension",
            "heart_disease",
        ],

        "Clinical Measurements": [
            "HbA1c_level",
            "blood_glucose_level",
        ],
    },

    "heart": {

        "Lifestyle & Patient Profile": [
            "Sex",
            "AgeCategory",
            "PhysicalActivities",
            "SleepHours",
            "SmokerStatus",
            "AlcoholDrinkers",
            "BMI",
        ],

        "General Health": [
            "GeneralHealth",
            "PhysicalHealthDays",
            "MentalHealthDays",
        ],

        "Medical History": [
            "HadStroke",
            "HadAsthma",
            "HadKidneyDisease",
            "HadArthritis",
            "HadDiabetes",
        ],
    },

    "kidney": {

        "Lifestyle & Patient Profile": [
            "Age",
            "Gender",
            "BMI",
            "Smoking",
            "AlcoholConsumption",
            "PhysicalActivity",
            "DietQuality",
            "SleepQuality",
        ],

        "Family & Medical History": [
            "FamilyHistoryKidneyDisease",
            "FamilyHistoryHypertension",
            "FamilyHistoryDiabetes",
            "Edema",
        ],

        "Clinical Measurements": [
            "SystolicBP",
            "DiastolicBP",
            "FastingBloodSugar",
            "HbA1c",
            "SerumCreatinine",
            "BUNLevels",
            "GFR",
            "ProteinInUrine",
            "ACR",
            "HemoglobinLevels",
        ],
    },
}


GROUP_DESCRIPTIONS = {

    "Lifestyle & Patient Profile":
        "Demographic information and everyday lifestyle factors.",

    "General Health":
        "Overall physical and mental health indicators.",

    "Medical History":
        "Previous conditions relevant to this assessment.",

    "Family & Medical History":
        "Family history and current medical indicators.",

    "Clinical Measurements":
        "Laboratory results and measurable clinical values.",
}


# =============================================================================
# FRIENDLY FIELD LABELS
# =============================================================================

FIELD_LABELS = {

    "gender": "Gender",
    "age": "Age",
    "hypertension": "History of Hypertension",
    "heart_disease": "History of Heart Disease",
    "smoking_history": "Smoking History",
    "bmi": "Body Mass Index (BMI)",
    "HbA1c_level": "HbA1c Level (%)",
    "blood_glucose_level": "Blood Glucose Level (mg/dL)",

    "Sex": "Sex",
    "GeneralHealth": "General Health",
    "PhysicalHealthDays": "Poor Physical Health Days",
    "MentalHealthDays": "Poor Mental Health Days",
    "PhysicalActivities": "Regular Physical Activity",
    "SleepHours": "Average Sleep Hours",
    "HadStroke": "History of Stroke",
    "HadAsthma": "History of Asthma",
    "HadKidneyDisease": "History of Kidney Disease",
    "HadArthritis": "History of Arthritis",
    "HadDiabetes": "History of Diabetes",
    "SmokerStatus": "Smoking Status",
    "AgeCategory": "Age Category",
    "BMI": "Body Mass Index (BMI)",
    "AlcoholDrinkers": "Alcohol Consumption Status",

    "Age": "Age",
    "Gender": "Gender",
    "Smoking": "Smoking",
    "AlcoholConsumption": "Alcohol Consumption",
    "PhysicalActivity": "Physical Activity",
    "DietQuality": "Diet Quality Score",
    "SleepQuality": "Sleep Quality Score",
    "FamilyHistoryKidneyDisease":
        "Family History of Kidney Disease",
    "FamilyHistoryHypertension":
        "Family History of Hypertension",
    "FamilyHistoryDiabetes":
        "Family History of Diabetes",
    "SystolicBP": "Systolic Blood Pressure (mmHg)",
    "DiastolicBP": "Diastolic Blood Pressure (mmHg)",
    "FastingBloodSugar": "Fasting Blood Sugar (mg/dL)",
    "HbA1c": "HbA1c (%)",
    "SerumCreatinine": "Serum Creatinine",
    "BUNLevels": "Blood Urea Nitrogen (BUN)",
    "GFR": "Estimated GFR",
    "ProteinInUrine": "Protein in Urine",
    "ACR": "Albumin-to-Creatinine Ratio (ACR)",
    "HemoglobinLevels": "Hemoglobin Level",
    "Edema": "Presence of Edema",
}


BINARY_FIELDS = {
    "hypertension",
    "heart_disease",
    "Smoking",
    "FamilyHistoryKidneyDisease",
    "FamilyHistoryHypertension",
    "FamilyHistoryDiabetes",
    "Edema",
}


# =============================================================================
# LOAD RESOURCES
# =============================================================================

@st.cache_resource
def load_model(name):

    with open(
        MODELS_DIR / f"{name}_model.pkl",
        "rb",
    ) as file:

        return pickle.load(file)


@st.cache_resource
def load_named_model(disease, model_name):
    """Load a benchmark model saved by train_model.py; fall back to the recommended model."""
    safe_name = model_name.lower().replace(" ", "_")
    path = MODELS_DIR / "all_models" / f"{disease}_{safe_name}.pkl"
    if path.exists():
        with open(path, "rb") as file:
            return pickle.load(file)
    if model_name == meta[disease]["best_model"]:
        return MODELS[disease]
    return None

def available_models(disease):
    return [row["model"] for row in meta[disease].get("validation_model_results", [])]

def model_reason(disease):
    reasons = {
        "diabetes": "Gradient Boosting is recommended because it achieved the strongest validation macro-F1 score for the diabetes dataset.",
        "heart": "Logistic Regression is recommended because AHEAD prioritizes sensitivity in cardiovascular screening and selected the model using the F2 score, which gives more weight to recall.",
        "kidney": "XGBoost is recommended because it achieved the strongest validation macro-F1 score for the kidney dataset.",
    }
    return reasons[disease]


@st.cache_data
def load_metadata():

    with open(
        MODELS_DIR / "model_meta.json",
        "r",
    ) as file:

        return json.load(file)


@st.cache_data
def load_dataset(name):

    return pd.read_csv(
        DATA_DIR / DATASET_FILES[name]
    )


meta = load_metadata()

MODELS = {
    "diabetes": load_model("diabetes"),
    "heart": load_model("heart"),
    "kidney": load_model("kidney"),
}


# =============================================================================
# META HELPERS
# =============================================================================

def final_metrics(name):

    return meta[name].get(
        "final_test_metrics",
        {},
    )


def test_accuracy(name):

    return float(
        final_metrics(name).get(
            "accuracy",
            0,
        )
    )


def test_macro_f1(name):

    return float(
        final_metrics(name).get(
            "macro_f1",
            0,
        )
    )


def disease_recall(name):

    return float(
        final_metrics(name).get(
            "recall_disease",
            0,
        )
    )


def roc_auc(name):

    value = final_metrics(name).get(
        "roc_auc"
    )

    return (
        float(value)
        if value is not None
        else 0.0
    )


def decision_threshold(name):

    return float(
        meta[name].get(
            "decision_threshold",
            0.5,
        )
    )


def pretty_label(feature):

    return FIELD_LABELS.get(
        feature,
        feature.replace("_", " ").title(),
    )


# =============================================================================
# INPUT COMPONENTS
# =============================================================================

def render_feature_input(
    dataset,
    feature,
    key,
):

    series = dataset[
        feature
    ].dropna()

    label = pretty_label(
        feature
    )

    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):

        options = (
            series
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if feature != "AgeCategory":
            options = sorted(options)

        return st.selectbox(
            label,
            options,
            key=key,
        )

    unique_values = sorted(
        series.unique().tolist()
    )

    if (
        feature == "Gender"
        and set(unique_values).issubset(
            {0, 1}
        )
    ):

        return st.selectbox(
            label,
            [0, 1],
            format_func=lambda x:
                "Male"
                if x == 0
                else "Female",
            key=key,
        )

    if (
        feature in BINARY_FIELDS
        and set(unique_values).issubset(
            {0, 1}
        )
    ):

        return st.selectbox(
            label,
            [0, 1],
            format_func=lambda x:
                "No"
                if x == 0
                else "Yes",
            key=key,
        )

    minimum = float(
        series.min()
    )

    maximum = float(
        series.max()
    )

    median = float(
        series.median()
    )

    if (
        feature in {"age", "Age"}
        or pd.api.types.is_integer_dtype(
            series
        )
    ):

        return st.number_input(
            label,
            min_value=int(
                np.floor(minimum)
            ),
            max_value=int(
                np.ceil(maximum)
            ),
            value=int(
                round(median)
            ),
            step=1,
            key=key,
        )

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=median,
        step=0.1,
        key=key,
    )


# =============================================================================
# GAUGE
# =============================================================================

def risk_gauge(
    probability,
    threshold,
    title,
):

    probability_percent = (
        probability * 100
    )

    threshold_percent = (
        threshold * 100
    )

    figure = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=round(
                probability_percent,
                1,
            ),

            number={
                "suffix": "%",
                "font": {
                    "size": 38,
                    "color": "#17324D",
                },
            },

            title={
                "text": title,
                "font": {
                    "size": 14,
                    "color": "#667085",
                },
            },

            gauge={

                "axis": {
                    "range": [0, 100],
                },

                "bar": {
                    "color": "#177D83",
                    "thickness": 0.28,
                },

                "steps": [

                    {
                        "range": [
                            0,
                            threshold_percent,
                        ],
                        "color": "#E8F5EF",
                    },

                    {
                        "range": [
                            threshold_percent,
                            100,
                        ],
                        "color": "#FBEDED",
                    },
                ],

                "threshold": {

                    "line": {
                        "color": "#B34D4D",
                        "width": 4,
                    },

                    "value":
                        threshold_percent,
                },
            },
        )
    )

    figure.update_layout(
        height=275,

        margin=dict(
            l=25,
            r=25,
            t=55,
            b=15,
        ),

        paper_bgcolor="rgba(0,0,0,0)",
    )

    return figure


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def model_feature_importance(
    model,
    title,
    top_n=12,
):

    classifier = model.named_steps[
        "classifier"
    ]

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        feature_names = [

            feature.split(
                "__",
                1
            )[-1]

            for feature
            in feature_names
        ]

    except Exception:

        return None

    if hasattr(
        classifier,
        "feature_importances_"
    ):

        importance = (
            classifier
            .feature_importances_
        )

        axis_label = (
            "Relative model importance"
        )

    elif hasattr(
        classifier,
        "coef_"
    ):

        importance = np.abs(
            classifier.coef_[0]
        )

        axis_label = (
            "Absolute coefficient magnitude"
        )

    else:

        return None

    if (
        len(feature_names)
        != len(importance)
    ):

        return None

    frame = pd.DataFrame(
        {
            "Feature":
                feature_names,

            "Importance":
                importance,
        }
    )

    frame = (
        frame
        .sort_values(
            "Importance",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    figure = px.bar(
        frame,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
    )

    figure.update_traces(
        marker_color="#277E87"
    )

    figure.update_layout(

        height=410,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        xaxis_title=axis_label,

        yaxis_title="",

        title_font=dict(
            size=15,
            color="#17324D",
        ),
    )

    figure.update_xaxes(
        gridcolor="#EDF1F5"
    )

    return figure


# =============================================================================
# RESULT CARD
# =============================================================================

def render_result_card(
    prediction,
    probability,
    disease_label,
):

    percentage = (
        probability * 100
    )

    if prediction == 1:

        st.html(
            f"""
<div class="result-card result-elevated">

    <div class="result-kicker">
        Screening Result
    </div>

    <div class="result-high-title">
        Elevated Risk Pattern
    </div>

    <div class="result-score">
        Estimated probability:
        <strong>{percentage:.1f}%</strong>
    </div>

    <div class="result-text">
        The screening model identified a pattern associated with
        elevated {disease_label} risk.
        This result is intended for early awareness
        and does not confirm that you have the condition.
    </div>

</div>
"""
        )

    else:

        st.html(
            f"""
<div class="result-card result-lower">

    <div class="result-kicker">
        Screening Result
    </div>

    <div class="result-low-title">
        Lower Predicted Risk
    </div>

    <div class="result-score">
        Estimated probability:
        <strong>{percentage:.1f}%</strong>
    </div>

    <div class="result-text">
        Your entered information did not meet the model's
        classification level for elevated {disease_label} risk.
        This does not rule out disease or replace routine screening.
    </div>

</div>
"""
        )


# =============================================================================
# RISK FACTORS FROM USER INPUT
# =============================================================================

def identify_risk_factors(
    disease,
    values,
):

    factors = []

    if disease == "diabetes":

        if float(
            values.get(
                "bmi",
                0,
            )
        ) >= 30:

            factors.append(
                "Higher BMI"
            )

        if int(
            values.get(
                "hypertension",
                0,
            )
        ) == 1:

            factors.append(
                "History of hypertension"
            )

        if int(
            values.get(
                "heart_disease",
                0,
            )
        ) == 1:

            factors.append(
                "History of heart disease"
            )

        if float(
            values.get(
                "HbA1c_level",
                0,
            )
        ) >= 5.7:

            factors.append(
                "Higher entered HbA1c"
            )

        if float(
            values.get(
                "blood_glucose_level",
                0,
            )
        ) >= 100:

            factors.append(
                "Higher entered glucose"
            )

        smoking = str(
            values.get(
                "smoking_history",
                ""
            )
        ).lower()

        if (
            "current" in smoking
            or "former" in smoking
            or "ever" in smoking
        ):

            factors.append(
                "Smoking history"
            )

    elif disease == "heart":

        if float(
            values.get(
                "BMI",
                0,
            )
        ) >= 30:

            factors.append(
                "Higher BMI"
            )

        if str(
            values.get(
                "PhysicalActivities",
                ""
            )
        ).lower() == "no":

            factors.append(
                "No regular physical activity"
            )

        smoking = str(
            values.get(
                "SmokerStatus",
                ""
            )
        ).lower()

        if (
            "current" in smoking
            or "former" in smoking
        ):

            factors.append(
                "Smoking history"
            )

        if str(
            values.get(
                "HadStroke",
                ""
            )
        ).lower() == "yes":

            factors.append(
                "Previous stroke"
            )

        if str(
            values.get(
                "HadDiabetes",
                ""
            )
        ).lower() == "yes":

            factors.append(
                "History of diabetes"
            )

        if str(
            values.get(
                "HadKidneyDisease",
                ""
            )
        ).lower() == "yes":

            factors.append(
                "History of kidney disease"
            )

    elif disease == "kidney":

        if int(
            values.get(
                "Smoking",
                0,
            )
        ) == 1:

            factors.append(
                "Smoking"
            )

        if float(
            values.get(
                "BMI",
                0,
            )
        ) >= 30:

            factors.append(
                "Higher BMI"
            )

        if int(
            values.get(
                "FamilyHistoryKidneyDisease",
                0,
            )
        ) == 1:

            factors.append(
                "Family history of kidney disease"
            )

        if int(
            values.get(
                "FamilyHistoryHypertension",
                0,
            )
        ) == 1:

            factors.append(
                "Family history of hypertension"
            )

        if int(
            values.get(
                "FamilyHistoryDiabetes",
                0,
            )
        ) == 1:

            factors.append(
                "Family history of diabetes"
            )

        if int(
            values.get(
                "Edema",
                0,
            )
        ) == 1:

            factors.append(
                "Reported edema"
            )

        if float(
            values.get(
                "GFR",
                100,
            )
        ) < 60:

            factors.append(
                "Lower entered GFR"
            )

        if float(
            values.get(
                "ACR",
                0,
            )
        ) >= 30:

            factors.append(
                "Higher entered urine ACR"
            )

    return factors


# =============================================================================
# NEXT STEPS
# =============================================================================

def get_next_steps(
    disease,
    prediction,
    values,
):

    steps = []

    # -------------------------------------------------------------------------
    # DIABETES
    # -------------------------------------------------------------------------

    if disease == "diabetes":

        if prediction == 1:

            steps = [

                (
                    "Arrange a medical review",
                    "Book an appointment with a primary-care doctor "
                    "or diabetes clinic to review this result together "
                    "with your symptoms, medical history and previous tests."
                ),

                (
                    "Ask about confirmatory blood testing",
                    "Discuss whether HbA1c, fasting plasma glucose or "
                    "other appropriate diabetes testing should be repeated "
                    "or confirmed clinically."
                ),

                (
                    "Review blood pressure and cardiovascular risk",
                    "Diabetes risk commonly overlaps with blood-pressure, "
                    "cholesterol and cardiovascular risk, so ask whether "
                    "these should also be checked."
                ),

                (
                    "Make a structured lifestyle plan",
                    "If your clinician confirms prediabetes or increased risk, "
                    "work on sustainable nutrition, activity and weight goals "
                    "rather than extreme diets or rapid weight loss."
                ),
            ]

        else:

            steps = [

                (
                    "Continue routine screening",
                    "A lower screening result does not rule out diabetes. "
                    "Continue blood-sugar screening when recommended based "
                    "on your age, medical history and other risk factors."
                ),

                (
                    "Maintain metabolic health",
                    "Continue regular physical activity, balanced meals, "
                    "adequate sleep and realistic long-term weight management."
                ),
            ]

    # -------------------------------------------------------------------------
    # HEART
    # -------------------------------------------------------------------------

    elif disease == "heart":

        if prediction == 1:

            steps = [

                (
                    "Arrange a cardiovascular risk review",
                    "Book a primary-care appointment and discuss whether "
                    "a cardiology assessment is appropriate, especially if "
                    "you have previous stroke, diabetes, kidney disease or "
                    "a significant smoking history."
                ),

                (
                    "Check major cardiovascular risk factors",
                    "Ask for review of blood pressure, cholesterol or lipid "
                    "levels, blood sugar and your overall cardiovascular risk."
                ),

                (
                    "Address smoking if applicable",
                    "If you currently smoke, ask your healthcare professional "
                    "about evidence-based smoking-cessation support."
                ),

                (
                    "Build physical activity gradually and safely",
                    "If you do not have medical restrictions, work toward "
                    "regular physical activity. If exercise causes chest "
                    "discomfort, unusual breathlessness or dizziness, stop "
                    "and seek medical advice."
                ),
            ]

        else:

            steps = [

                (
                    "Keep monitoring cardiovascular health",
                    "A lower screening result does not eliminate future heart "
                    "risk. Continue routine checks of blood pressure, "
                    "cholesterol and blood sugar when recommended."
                ),

                (
                    "Protect modifiable risk factors",
                    "Avoid smoking, stay physically active, maintain healthy "
                    "sleep and follow a balanced eating pattern."
                ),
            ]

    # -------------------------------------------------------------------------
    # KIDNEY
    # -------------------------------------------------------------------------

    elif disease == "kidney":

        if prediction == 1:

            steps = [

                (
                    "Arrange a kidney-health review",
                    "Book an appointment with a primary-care doctor. "
                    "Depending on your laboratory results and medical history, "
                    "they may recommend referral to a nephrologist."
                ),

                (
                    "Ask about kidney-function testing",
                    "Discuss repeat kidney assessment including serum "
                    "creatinine/eGFR and a urine albumin-to-creatinine ratio "
                    "(uACR), especially if previous results were abnormal."
                ),

                (
                    "Review blood pressure and diabetes control",
                    "High blood pressure and diabetes can affect kidney health. "
                    "Ask whether your blood pressure, HbA1c or glucose needs "
                    "additional monitoring."
                ),

                (
                    "Review medicines and supplements",
                    "Tell your clinician or pharmacist about prescription "
                    "medicines, over-the-counter painkillers and supplements. "
                    "Do not stop prescribed medicine on your own."
                ),
            ]

        else:

            steps = [

                (
                    "Continue routine kidney monitoring",
                    "If you have diabetes, hypertension or a strong family "
                    "history of kidney disease, ask how often kidney-function "
                    "and urine testing should be performed."
                ),

                (
                    "Protect long-term kidney health",
                    "Maintain blood-pressure and blood-sugar control, avoid "
                    "smoking and discuss regular use of over-the-counter "
                    "painkillers with a healthcare professional."
                ),
            ]

    return steps


# =============================================================================
# AHEAD INSIGHT
# =============================================================================

def generate_ai_guidance(
    disease,
    prediction,
    probability,
    risk_factors,
    user_values,
):

    disease_names = {
        "diabetes": "diabetes",
        "heart": "cardiovascular disease",
        "kidney": "chronic kidney disease",
    }

    result_label = (
        "Elevated screening risk"
        if prediction == 1
        else "Lower screening risk"
    )

    # Use friendly field labels instead of raw dataset column names.
    friendly_values = {
        pretty_label(key): value
        for key, value in user_values.items()
    }

    prompt = f"""
You are AHEAD Insight, the personalized interpretation layer inside AHEAD.

AHEAD is a university educational early-awareness screening prototype.
It is not a diagnostic medical device and must never present a screening
result as a confirmed diagnosis.

The disease-specific machine-learning model has already produced the screening
result. Your role is NOT to make a new prediction and NOT to repeat AHEAD's
standard Recommended Next Steps. Your role is to help the user understand the
existing result in the context of the information they entered.

Condition screened: {disease_names.get(disease, disease)}
Screening result: {result_label}
Model screening score: {probability * 100:.1f}%
Relevant factors identified by AHEAD: {risk_factors if risk_factors else 'No major rule-based factors highlighted'}
Patient-entered information: {friendly_values}

Write a concise, calm, professional and patient-friendly interpretation.

Use exactly these headings:
### Understanding Your Result
### What Stands Out From Your Inputs
### Questions to Discuss With a Healthcare Professional

Requirements:
- Explain the screening result in simple language and make clear that it is an educational screening estimate, not a diagnosis.
- Explain the model score without describing it as the user's true medical probability.
- Relate the explanation to the user's actual entered information.
- In "What Stands Out From Your Inputs", identify 2 to 4 relevant entered values or history items that are worth understanding or discussing.
- Do not claim that any single input caused the model prediction.
- If an individual clinical measurement may deserve professional interpretation even when the overall screening result is lower, clearly explain that distinction.
- In "Questions to Discuss With a Healthcare Professional", generate 2 to 4 specific questions based on the screening result and entered information.
- Do NOT create another "Recommended Next Steps" section and do not repeat generic lifestyle advice already covered elsewhere in AHEAD.
- Do not diagnose the user or say they definitely have or definitely do not have the condition.
- Do not recommend starting, stopping, or changing prescription medication.
- Do not invent symptoms, diagnoses, laboratory results, or risk factors that were not provided.
- Keep the answer brief enough to read comfortably inside a screening application.
"""

    gemini_client = genai.Client()
    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
    )

    return response.text


# =============================================================================
# URGENT CARE INFORMATION
# =============================================================================

def urgent_message(
    disease
):

    if disease == "heart":

        return (
            "<strong>Seek emergency medical care immediately</strong> "
            "for new or severe chest pressure or pain, shortness of breath, "
            "fainting, or pain/discomfort spreading to the arm, back, "
            "neck or jaw. Do not rely on the AHEAD result during an emergency."
        )

    if disease == "kidney":

        return (
            "<strong>Seek urgent medical care</strong> if you develop severe "
            "shortness of breath, confusion, very little or no urine, rapidly "
            "worsening swelling, or another severe new symptom. "
            "This screening tool is not designed for emergencies."
        )

    return (
        "<strong>Seek urgent medical care</strong> for severe illness, "
        "confusion, fainting, persistent vomiting, severe dehydration or "
        "other rapidly worsening symptoms. Do not use this screening result "
        "to decide whether emergency care is needed."
    )


# =============================================================================
# PREDICTOR
# =============================================================================

def render_predictor(
    disease,
    title,
    subtitle,
    disease_label,
):

    recommended_model = meta[disease]["best_model"]
    choices = available_models(disease)
    if recommended_model in choices:
        choices = [recommended_model] + [name for name in choices if name != recommended_model]

    st.markdown("#### Prediction Model")
    selected_model_name = st.selectbox(
        "Choose the machine-learning model",
        choices,
        format_func=lambda name: f"{name} — Recommended" if name == recommended_model else name,
        key=f"{disease}_model_selector",
    )
    model = load_named_model(disease, selected_model_name)
    if model is None:
        st.warning(
            f"{selected_model_name} has not been saved yet. Run `python train_model.py` once after this upgrade. "
            f"AHEAD is using {recommended_model} for now."
        )
        selected_model_name = recommended_model
        model = MODELS[disease]

    with st.expander("Why does AHEAD recommend this model?"):
        st.write(model_reason(disease))
        st.caption("Alternative models are available for comparison and research. The recommended model remains the default.")

    dataset = load_dataset(
        disease
    )

    features = meta[
        disease
    ][
        "features"
    ]

    # Important:
    # threshold remains active internally, but is intentionally
    # not displayed in the patient-facing interface.
    threshold = (
        decision_threshold(disease)
        if selected_model_name == recommended_model
        else 0.5
    )

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------

    st.html(
        f"""
<div class="clinical-header">

    <div class="header-eyebrow">
        AHEAD Clinical Screening
    </div>

    <h1>{title}</h1>

    <p>{subtitle}</p>

</div>
"""
    )

    # -------------------------------------------------------------------------
    # FORM
    # -------------------------------------------------------------------------

    with st.form(
        f"{disease}_form"
    ):

        values = {}

        for section_number, (
            section_name,
            section_features,
        ) in enumerate(
            FEATURE_GROUPS[
                disease
            ].items(),
            start=1,
        ):

            st.html(
                f"""
<div class="form-section-title">

    <div class="form-section-number">
        {section_number}
    </div>

    <div>

        <h3>
            {section_name}
        </h3>

        <p>
            {GROUP_DESCRIPTIONS[section_name]}
        </p>

    </div>

</div>
"""
            )

            active_features = [

                feature

                for feature
                in section_features

                if feature
                in features
            ]

            columns = st.columns(
                3
            )

            for index, feature in enumerate(
                active_features
            ):

                with columns[
                    index % 3
                ]:

                    values[
                        feature
                    ] = render_feature_input(

                        dataset,

                        feature,

                        f"{disease}_{feature}",
                    )

        submitted = (
            st.form_submit_button(
                "Generate Screening Result",
                use_container_width=True,
            )
        )

    # -------------------------------------------------------------------------
    # RESULTS
    # -------------------------------------------------------------------------

    if submitted:

        input_frame = pd.DataFrame(
            [values],
            columns=features,
        )

        probability = float(
            model.predict_proba(
                input_frame
            )[0][1]
        )

        prediction = int(
            probability
            >= threshold
        )

        st.write("")

        st.html(
            """
<div class="section-heading">

    <h2>
        Screening Result
    </h2>

    <p>
        Risk estimate generated from the information entered above.
    </p>

</div>
"""
        )

        result_column, gauge_column = (
            st.columns(
                [1, 1]
            )
        )

        with result_column:

            render_result_card(
                prediction,
                probability,
                disease_label,
            )

        with gauge_column:

            st.plotly_chart(

                risk_gauge(
                    probability,
                    threshold,
                    f"{disease_label.title()} Probability",
                ),

                use_container_width=True,
            )

        # ---------------------------------------------------------------------
        # NEXT STEPS
        # ---------------------------------------------------------------------

        st.html(
            """
<div class="section-heading">

    <h2>
        Recommended Next Steps
    </h2>

    <p>
        Practical actions to consider after this screening result.
    </p>

</div>
"""
        )

        next_steps = get_next_steps(
            disease,
            prediction,
            values,
        )

        for index, (
            step_title,
            step_text,
        ) in enumerate(
            next_steps,
            start=1,
        ):

            st.html(
                f"""
<div class="action-card">

    <span class="action-number">
        {index}
    </span>

    <span class="action-title">
        {step_title}
    </span>

    <div class="action-text">
        {step_text}
    </div>

</div>
"""
            )

        # ---------------------------------------------------------------------
        # RISK FACTORS
        # ---------------------------------------------------------------------

        risk_factors = (
            identify_risk_factors(
                disease,
                values,
            )
        )

        st.html(
            """
<div class="section-heading">

    <h2>
        Factors Identified From Your Inputs
    </h2>

    <p>
        Entered factors that may be relevant to the screening result.
    </p>

</div>
"""
        )

        if risk_factors:

            for factor in risk_factors:

                st.markdown(
                    f"- **{factor}**"
                )

        else:

            st.info(
                "No major rule-based risk factors were highlighted "
                "from the entered values. The machine-learning model "
                "may still use combinations of features in its prediction."
            )

        # ---------------------------------------------------------------------
        # URGENT CARE
        # ---------------------------------------------------------------------

        st.html(
            f"""
<div class="urgent-box">
    {urgent_message(disease)}
</div>
"""
        )

        # ---------------------------------------------------------------------
        # AHEAD INSIGHT
        # ---------------------------------------------------------------------

        st.html(
            """
<div class="section-heading">

    <h2>
        AHEAD Insight
    </h2>

    <p>
        Personalized explanation of your screening.
    </p>

</div>
"""
        )

        with st.spinner(
            "Preparing your personalized insight..."
        ):

            try:

                ai_guidance = generate_ai_guidance(
                    disease=disease,
                    prediction=prediction,
                    probability=probability,
                    risk_factors=risk_factors,
                    user_values=values,
                )

                st.markdown(
                    ai_guidance
                )

            except Exception as error:

                st.warning(
                    "AHEAD Insight is temporarily unavailable. "
                    "Your screening result and the standard AHEAD recommendations "
                    "above are still available."
                )

        # ---------------------------------------------------------------------
        # MODEL INTERPRETATION
        # ---------------------------------------------------------------------

        st.html(
            """
<div class="section-heading">

    <h2>
        Model Interpretation
    </h2>

    <p>
        Features with the strongest overall influence on this model.
    </p>

</div>
"""
        )

        importance_figure = (
            model_feature_importance(

                model,

                f"{disease_label.title()} — Leading Model Features",
            )
        )

        if importance_figure is not None:

            st.plotly_chart(
                importance_figure,
                use_container_width=True,
            )

            if (
                meta[disease][
                    "best_model"
                ]
                == "LogisticRegression"
            ):

                st.caption(
                    "For Logistic Regression, this chart uses absolute "
                    "coefficient magnitude. It represents model influence, "
                    "not medical causation."
                )

        # ---------------------------------------------------------------------
        # INPUT SUMMARY
        # ---------------------------------------------------------------------

        with st.expander(
            "View entered information"
        ):

            summary = (
                input_frame
                .T
                .rename(
                    columns={
                        0:
                            "Entered Value"
                    }
                )
            )

            summary.index = [

                pretty_label(
                    feature
                )

                for feature
                in summary.index
            ]

            st.dataframe(
                summary,
                use_container_width=True,
            )

    st.html(
        """
<div class="disclaimer">

<strong>Medical disclaimer:</strong>
AHEAD is an educational and research screening prototype.
It does not diagnose disease and should not be used to start,
stop or change treatment. A qualified healthcare professional
should interpret symptoms and clinical test results.

</div>
"""
    )


# =============================================================================
# AUTHENTICATION, CLINICAL DASHBOARD & CHATBOT
# =============================================================================

DEMO_ACCOUNTS = {
    "user@ahead.demo": {"password": "user123", "role": "User", "name": "AHEAD User"},
    "doctor@ahead.demo": {"password": "doctor123", "role": "Doctor/Admin", "name": "Dr. AHEAD"},
}

def render_login():
    st.html("""
<div class="home-hero">
    <div class="home-eyebrow">Secure Access Portal</div>
    <h1>AHEAD</h1>
    <p>Advanced Health Early Awareness and Disease Detection System</p>
</div>
""")
    left, center, right = st.columns([1, 1.25, 1])
    with center:
        st.markdown("### Sign in")
        st.caption("Choose the appropriate account to access AHEAD.")
        email = st.text_input("Email", placeholder="name@example.com")
        password = st.text_input("Password", type="password")
        if st.button("Log In", use_container_width=True):
            account = DEMO_ACCOUNTS.get(email.strip().lower())
            if account and password == account["password"]:
                st.session_state.authenticated = True
                st.session_state.role = account["role"]
                st.session_state.display_name = account["name"]
                st.rerun()
            else:
                st.error("Incorrect email or password.")
        with st.expander("Demo accounts"):
            st.code("User: user@ahead.demo / user123\nDoctor: doctor@ahead.demo / doctor123")
            st.caption("These credentials are for the university prototype only. Production deployment requires secure authentication and hashed passwords.")

def read_patient_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def render_clinical_dashboard():
    st.html("""
<div class="clinical-header">
    <div class="header-eyebrow">Hospital Clinical Decision Support</div>
    <h1>Clinical Dashboard</h1>
    <p>Upload a disease-specific patient file, select one record and generate an AHEAD model screening result for clinical review.</p>
</div>
""")
    st.warning("AHEAD is a research and educational decision-support prototype. Its output is not a confirmed diagnosis and must be interpreted by a qualified clinician.")
    disease_labels = {"diabetes":"Diabetes", "heart":"Cardiovascular Disease", "kidney":"Chronic Kidney Disease"}
    disease = st.selectbox("Disease screening", list(disease_labels), format_func=lambda x: disease_labels[x])
    recommended = meta[disease]["best_model"]
    choices = available_models(disease)
    if recommended in choices:
        choices = [recommended] + [x for x in choices if x != recommended]
    selected_name = st.selectbox("Prediction model", choices, format_func=lambda x: f"{x} — Recommended" if x == recommended else x, key="admin_model")
    with st.expander("Why is this model recommended?"):
        st.write(model_reason(disease))
    uploaded = st.file_uploader("Upload patient records", type=["csv", "xlsx", "xls"], help="Upload one disease-specific file. Each row represents one patient record.")
    if uploaded is None:
        st.info("Upload a CSV or Excel file to begin.")
        return
    try:
        patient_df = read_patient_file(uploaded)
    except Exception as error:
        st.error(f"AHEAD could not read this file: {error}")
        return
    required = meta[disease]["features"]
    missing = [col for col in required if col not in patient_df.columns]
    if missing:
        st.error("The uploaded file is missing required columns: " + ", ".join(missing))
        st.caption("Required columns: " + ", ".join(required))
        return
    st.success(f"File loaded successfully — {len(patient_df):,} patient records found.")
    preview_cols = [c for c in ["PatientID", "patient_id", "ID", "id"] if c in patient_df.columns]
    preview_cols += [c for c in required if c not in preview_cols][:6]
    st.dataframe(patient_df[preview_cols], use_container_width=True, height=300)
    labels = []
    id_col = preview_cols[0] if preview_cols and preview_cols[0] not in required else None
    for i in range(len(patient_df)):
        labels.append(f"Row {i + 1}" + (f" — {patient_df.iloc[i][id_col]}" if id_col else ""))
    selected_label = st.selectbox("Select patient record", labels)
    row_index = labels.index(selected_label)
    selected = patient_df.iloc[[row_index]][required].copy()
    with st.expander("Selected patient information"):
        summary = selected.T.rename(columns={selected.index[0]: "Value"})
        summary.index = [pretty_label(x) for x in summary.index]
        st.dataframe(summary, use_container_width=True)
    if st.button("Run Clinical Screening", use_container_width=True):
        model = load_named_model(disease, selected_name)
        if model is None:
            st.error(f"The saved {selected_name} pipeline is not available yet. Run `python train_model.py` after installing the upgraded files.")
            return
        threshold = decision_threshold(disease) if selected_name == recommended else 0.5
        probability = float(model.predict_proba(selected)[0][1])
        prediction = int(probability >= threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected model", selected_name)
        c2.metric("Model score", f"{probability*100:.1f}%")
        c3.metric("Screening classification", "Flagged for review" if prediction else "Not flagged")
        render_result_card(prediction, probability, disease_labels[disease])
        st.caption("The model score is an algorithmic screening output, not the patient's true probability of disease and not a confirmed diagnosis.")

def generate_chat_reply(message):
    prompt = f"""You are AHEAD Assistant inside a university healthcare screening prototype. Answer questions about AHEAD, diabetes, cardiovascular disease, chronic kidney disease, common screening measurements, and how to use the website. Give general educational information only. Do not diagnose a person, do not claim a screening result confirms disease, and do not recommend starting/stopping/changing prescription medication. For urgent symptoms advise appropriate urgent medical care. Keep answers concise and patient-friendly.\n\nUser question: {message}"""
    gemini_client = genai.Client()
    response = gemini_client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    return response.text

def render_chatbot():
    st.html("""
<div class="clinical-header">
    <div class="header-eyebrow">AHEAD Assistant</div>
    <h1>Ask AHEAD</h1>
    <p>Ask general questions about the three conditions, screening measurements, model results, or how to use AHEAD.</p>
</div>
""")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role":"assistant", "content":"Hi! I’m AHEAD Assistant. What would you like to know about AHEAD or its screening conditions?"}]
    for item in st.session_state.chat_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
    message = st.chat_input("Ask AHEAD a question...")
    if message:
        st.session_state.chat_messages.append({"role":"user", "content":message})
        with st.chat_message("user"):
            st.markdown(message)
        with st.chat_message("assistant"):
            try:
                answer = generate_chat_reply(message)
            except Exception:
                answer = "AHEAD Assistant is temporarily unavailable. The screening tools can still be used normally."
            st.markdown(answer)
        st.session_state.chat_messages.append({"role":"assistant", "content":answer})

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    render_login()
    st.stop()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.html(
        """
<div class="sidebar-brand">

    <div class="brand-symbol">
        ✚
    </div>

    <h2>
        AHEAD
    </h2>

    <p>
        Clinical Screening &<br>
        Early Awareness Platform
    </p>

</div>
"""
    )

    st.divider()

    st.caption(f"SIGNED IN AS · {st.session_state.get('role', 'User')}")
    st.markdown(f"**{st.session_state.get('display_name', 'AHEAD User')}**")

    if st.session_state.get("role") == "Doctor/Admin":
        nav_options = [
            "Clinical Dashboard",
            "Overview",
            "Diabetes Screening",
            "Cardiovascular Screening",
            "Kidney Health Screening",
            "AHEAD Assistant",
            "Data & Analytics",
            "About AHEAD",
        ]
    else:
        nav_options = [
            "Overview",
            "Diabetes Screening",
            "Cardiovascular Screening",
            "Kidney Health Screening",
            "AHEAD Assistant",
            "Data & Analytics",
            "About AHEAD",
        ]

    page = st.radio(
        "Navigation",
        nav_options,
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        "EARLY AWARENESS SCREENING"
    )

    st.caption(
        "Educational screening for diabetes, cardiovascular "
        "and kidney health."
    )

    st.divider()
    if st.button("Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.pop("role", None)
        st.session_state.pop("display_name", None)
        st.rerun()


# =============================================================================
# OVERVIEW
# =============================================================================

if page == "Clinical Dashboard":
    render_clinical_dashboard()

elif page == "AHEAD Assistant":
    render_chatbot()

elif page == "Overview":

    diabetes_df = load_dataset(
        "diabetes"
    )

    heart_df = load_dataset(
        "heart"
    )

    kidney_df = load_dataset(
        "kidney"
    )

    total_records = (
        len(diabetes_df)
        + len(heart_df)
        + len(kidney_df)
    )

    st.html(
        """
<div class="home-hero">

    <div class="home-eyebrow">
        Clinical Intelligence Platform
    </div>

    <h1>
        AHEAD
    </h1>

    <p>
        Advanced Health Early Awareness and Disease Detection.
        A machine-learning screening platform supporting
        early awareness of diabetes, cardiovascular disease
        and chronic kidney disease.
    </p>

</div>
"""
    )

    # -------------------------------------------------------------------------
    # USER-FOCUSED OVERVIEW CARDS
    # -------------------------------------------------------------------------

    metric_columns = (
        st.columns(3)
    )

    cards = [

        (
            "3",
            "Screening Services",
            "Diabetes · Heart · Kidney"
        ),

        (
            f"{total_records:,}",
            "Health Records",
            "Across three screening datasets"
        ),

        (
            "3",
            "Health Conditions",
            "Early-awareness assessment"
        ),
    ]

    for column, (
        value,
        label,
        note,
    ) in zip(
        metric_columns,
        cards,
    ):

        with column:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        {label}
    </div>

    <div class="metric-value">
        {value}
    </div>

    <div class="metric-foot">
        {note}
    </div>

</div>
"""
            )

    # -------------------------------------------------------------------------
    # SCREENING SERVICES
    # -------------------------------------------------------------------------

    st.html(
        """
<div class="section-heading">

    <h2>
        Screening Services
    </h2>

    <p>
        Choose a condition from the navigation panel
        to begin an assessment.
    </p>

</div>
"""
    )

    columns = st.columns(
        3
    )

    with columns[0]:

        st.html(
            """
<div class="condition-card">

    <div class="condition-icon">
        D
    </div>

    <h3>
        Diabetes Risk Assessment
    </h3>

    <p>
        Assess diabetes risk using lifestyle information,
        medical history and clinical measurements such as
        HbA1c and blood glucose.
    </p>

    <span class="service-note">
        Lifestyle · Medical · Clinical
    </span>

</div>
"""
        )

    with columns[1]:

        st.html(
            """
<div class="condition-card">

    <div class="condition-icon">
        H
    </div>

    <h3>
        Cardiovascular Risk Assessment
    </h3>

    <p>
        Assess cardiovascular risk using lifestyle,
        general health and previous medical-history indicators.
    </p>

    <span class="service-note">
        Lifestyle · General Health · History
    </span>

</div>
"""
        )

    with columns[2]:

        st.html(
            """
<div class="condition-card">

    <div class="condition-icon">
        K
    </div>

    <h3>
        Kidney Health Assessment
    </h3>

    <p>
        Assess chronic kidney disease risk using lifestyle,
        family history and clinical kidney measurements.
    </p>

    <span class="service-note">
        Lifestyle · Family History · Clinical
    </span>

</div>
"""
        )

    # -------------------------------------------------------------------------
    # SIMPLE WORKFLOW
    # -------------------------------------------------------------------------

    st.html(
        """
<div class="section-heading">

    <h2>
        How AHEAD Works
    </h2>

    <p>
        Complete your health information and receive an
        early-awareness screening result with practical next steps.
    </p>

</div>
"""
    )

    workflow_columns = (
        st.columns(3)
    )

    workflow = [

        (
            "01",
            "Enter Your Information",
            "Complete the lifestyle, medical-history and clinical sections."
        ),

        (
            "02",
            "Generate Your Screening",
            "AHEAD analyses the information using the disease-specific screening system."
        ),

        (
            "03",
            "Review Your Next Steps",
            "See your result, relevant factors and practical recommendations."
        ),
    ]

    for column, (
        number,
        heading,
        description,
    ) in zip(
        workflow_columns,
        workflow,
    ):

        with column:

            st.html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        STEP {number}
    </div>

    <div style="
        margin-top:8px;
        color:#17324D;
        font-size:0.93rem;
        font-weight:700;
    ">
        {heading}
    </div>

    <div style="
        margin-top:6px;
        color:#7A8796;
        font-size:0.72rem;
        line-height:1.55;
    ">
        {description}
    </div>

</div>
"""
            )

    st.html(
        """
<div class="disclaimer">

<strong>Clinical notice:</strong>
AHEAD provides educational screening estimates only.
It does not provide medical diagnoses and does not
replace professional medical evaluation.

</div>
"""
    )


# =============================================================================
# SCREENING PAGES
# =============================================================================


    st.html("""
<div class="section-heading">
    <h2>User Experiences</h2>
    <p>Sample testimonials for the AHEAD prototype. Replace these with approved feedback collected from real users.</p>
</div>
""")
    t1, t2, t3 = st.columns(3)
    samples = [
        ("Easy to understand", "The screening flow made the information much easier to follow."),
        ("Clear health information", "I liked seeing the result together with the factors and questions to discuss with a professional."),
        ("Simple and organized", "The three screening services were straightforward to navigate."),
    ]
    for col, (heading, quote) in zip([t1, t2, t3], samples):
        with col:
            st.markdown(f"**{heading}**")
            st.caption(quote)
            st.caption("Sample testimonial - Demo only")

elif page == "Diabetes Screening":

    render_predictor(

        disease="diabetes",

        title="Diabetes Risk Assessment",

        subtitle=(
            "Complete the lifestyle, medical-history and clinical "
            "sections to generate an early-awareness diabetes "
            "screening estimate."
        ),

        disease_label="diabetes",
    )


elif page == "Cardiovascular Screening":

    render_predictor(

        disease="heart",

        title="Cardiovascular Risk Assessment",

        subtitle=(
            "Complete the lifestyle, general-health and medical-history "
            "sections to generate a cardiovascular screening estimate."
        ),

        disease_label="heart disease",
    )


elif page == "Kidney Health Screening":

    render_predictor(

        disease="kidney",

        title="Kidney Health Assessment",

        subtitle=(
            "Complete the lifestyle, family-history and clinical-laboratory "
            "sections to generate a chronic kidney disease screening estimate."
        ),

        disease_label="chronic kidney disease",
    )


# =============================================================================
# DATA & ANALYTICS
# =============================================================================

elif page == "Data & Analytics":

    st.html(
        """
<div class="clinical-header">

    <div class="header-eyebrow">
        AHEAD Analytics
    </div>

    <h1>
        Data & Model Explorer
    </h1>

    <p>
        Explore the datasets and review the technical
        performance of each production model.
    </p>

</div>
"""
    )

    diabetes_tab, heart_tab, kidney_tab = (
        st.tabs(
            [
                "Diabetes",
                "Cardiovascular",
                "Kidney",
            ]
        )
    )

    configurations = [

        (
            diabetes_tab,
            "diabetes",
            "Diabetes Dataset"
        ),

        (
            heart_tab,
            "heart",
            "Heart Disease Dataset"
        ),

        (
            kidney_tab,
            "kidney",
            "Kidney Disease Dataset"
        ),
    ]

    for (
        tab,
        disease,
        display_name,
    ) in configurations:

        with tab:

            dataset = load_dataset(
                disease
            )

            target = TARGET_COLUMNS[
                disease
            ]

            model_features = meta[
                disease
            ][
                "features"
            ]

            summary_cols = st.columns(
                4
            )

            summary_cols[0].metric(
                "Records",
                f"{len(dataset):,}"
            )

            summary_cols[1].metric(
                "Dataset Columns",
                len(dataset.columns)
            )

            summary_cols[2].metric(
                "Model Features",
                len(model_features)
            )

            summary_cols[3].metric(
                "Production Model",
                meta[disease][
                    "best_model"
                ]
            )

            st.divider()

            left, right = st.columns(
                2
            )

            with left:

                counts = (
                    dataset[
                        target
                    ]
                    .astype(str)
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "Class",
                    "Count"
                ]

                target_figure = px.bar(

                    counts,

                    x="Class",

                    y="Count",

                    title=(
                        f"{display_name} — "
                        f"Target Distribution"
                    ),
                )

                target_figure.update_traces(
                    marker_color="#287E86"
                )

                target_figure.update_layout(

                    height=330,

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(0,0,0,0)",

                    showlegend=False,
                )

                st.plotly_chart(
                    target_figure,
                    use_container_width=True,
                )

            with right:

                selected_feature = (
                    st.selectbox(

                        "Feature to explore",

                        model_features,

                        format_func=pretty_label,

                        key=(
                            f"explorer_"
                            f"{disease}"
                        ),
                    )
                )

                series = dataset[
                    selected_feature
                ]

                if (
                    pd.api.types
                    .is_numeric_dtype(
                        series
                    )
                    and series.nunique()
                    > 10
                ):

                    feature_figure = (
                        px.histogram(

                            dataset,

                            x=selected_feature,

                            nbins=35,

                            title=(
                                f"{pretty_label(selected_feature)} "
                                f"Distribution"
                            ),
                        )
                    )

                else:

                    feature_counts = (
                        series
                        .astype(str)
                        .value_counts()
                        .head(20)
                        .reset_index()
                    )

                    feature_counts.columns = [
                        "Value",
                        "Count"
                    ]

                    feature_figure = (
                        px.bar(

                            feature_counts,

                            x="Value",

                            y="Count",

                            title=(
                                f"{pretty_label(selected_feature)} "
                                f"Distribution"
                            ),
                        )
                    )

                feature_figure.update_traces(
                    marker_color="#5B8FA3"
                )

                feature_figure.update_layout(

                    height=330,

                    paper_bgcolor="rgba(0,0,0,0)",

                    plot_bgcolor="rgba(0,0,0,0)",

                    showlegend=False,
                )

                st.plotly_chart(
                    feature_figure,
                    use_container_width=True,
                )

            # -----------------------------------------------------------------
            # TECHNICAL MODEL INFORMATION BELONGS HERE
            # -----------------------------------------------------------------

            st.html(
                """
<div class="section-heading">

    <h2>
        Final Model Performance
    </h2>

    <p>
        Technical performance measured on the independent test set.
    </p>

</div>
"""
            )

            metrics = final_metrics(
                disease
            )

            metric_columns = st.columns(
                5
            )

            metric_columns[0].metric(
                "Accuracy",
                f"{metrics.get('accuracy', 0) * 100:.2f}%"
            )

            metric_columns[1].metric(
                "Disease Recall",
                f"{metrics.get('recall_disease', 0) * 100:.2f}%"
            )

            metric_columns[2].metric(
                "Disease F1",
                f"{metrics.get('f1_disease', 0):.4f}"
            )

            metric_columns[3].metric(
                "Macro-F1",
                f"{metrics.get('macro_f1', 0):.4f}"
            )

            auc = metrics.get(
                "roc_auc"
            )

            metric_columns[4].metric(
                "ROC-AUC",
                (
                    f"{auc:.4f}"
                    if auc is not None
                    else "N/A"
                )
            )

            # -----------------------------------------------------------------
            # ALGORITHM COMPARISON
            # -----------------------------------------------------------------

            all_results = meta[
                disease
            ].get(
                "all_model_results",
                []
            )

            if all_results:

                st.html(
                    """
<div class="section-heading">

    <h2>
        Algorithm Comparison
    </h2>

    <p>
        Comparison of the machine-learning approaches
        evaluated during model development.
    </p>

</div>
"""
                )

                comparison_df = pd.DataFrame(
                    all_results
                )

                columns_to_show = [
                    column
                    for column in [
                        "model",
                        "accuracy",
                        "precision_disease",
                        "recall_disease",
                        "f1_disease",
                        "macro_f1",
                        "f2_disease",
                        "roc_auc",
                    ]
                    if column
                    in comparison_df.columns
                ]

                comparison_df = comparison_df[
                    columns_to_show
                ]

                rename_map = {
                    "model": "Algorithm",
                    "accuracy": "Accuracy",
                    "precision_disease": "Disease Precision",
                    "recall_disease": "Disease Recall",
                    "f1_disease": "Disease F1",
                    "macro_f1": "Macro F1",
                    "f2_disease": "Disease F2",
                    "roc_auc": "ROC-AUC",
                }

                comparison_df = (
                    comparison_df
                    .rename(
                        columns=rename_map
                    )
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander(
                "View sample records"
            ):

                preview_columns = (

                    model_features
                    + [target]
                )

                preview_columns = [

                    column

                    for column
                    in preview_columns

                    if column
                    in dataset.columns
                ]

                st.dataframe(

                    dataset[
                        preview_columns
                    ].head(50),

                    use_container_width=True,
                )


# =============================================================================
# ABOUT
# =============================================================================

elif page == "About AHEAD":

    st.html(
        """
<div class="clinical-header">

    <div class="header-eyebrow">
        About the Project
    </div>

    <h1>
        About AHEAD
    </h1>

    <p>
        Advanced Health Early Awareness and Disease Detection
        is a Senior Design machine-learning project focused on
        early-awareness screening for chronic disease.
    </p>

</div>
"""
    )

    left, right = st.columns(
        2
    )

    with left:

        st.html(
            """
<div class="condition-card">

    <h3>
        What AHEAD Does
    </h3>

    <p>
        AHEAD accepts lifestyle, medical-history and clinical
        information and uses disease-specific machine-learning
        models to generate an early-awareness screening result.
    </p>

    <p>
        The current system includes diabetes, cardiovascular
        and chronic kidney disease assessments.
    </p>

</div>
"""
        )

    with right:

        st.html(
            """
<div class="condition-card">

    <h3>
        How Results Should Be Used
    </h3>

    <p>
        Results are designed for education and early awareness.
        They are not diagnoses and should not be used to make
        medication or treatment decisions.
    </p>

    <p>
        Concerning symptoms or abnormal laboratory results
        should be evaluated by a qualified healthcare professional.
    </p>

</div>
"""
        )

    st.html(
        """
<div class="disclaimer">

<strong>Medical disclaimer:</strong>
AHEAD is a university educational and research prototype.
It is not a medical device and is not intended for clinical
diagnosis, treatment selection or emergency decision-making.

</div>
"""
    )