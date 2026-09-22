"""
train_model.py
==============
Train and benchmark AHEAD machine-learning models using the new datasets.

For each disease, this script benchmarks:
    - Random Forest
    - Logistic Regression
    - Gradient Boosting
    - XGBoost (if installed)
    - SVM
    - Decision Tree

Selection:
    - Diabetes: best validation macro-F1
    - Heart: best validation F2 score for Disease
    - Kidney: best validation macro-F1

Heart threshold tuning:
    - The best heart model's probability threshold is tuned on the
      validation set only.
    - Final performance is reported once on the untouched test set.

Diseases:
1. Diabetes
2. Heart Disease
3. Chronic Kidney Disease

Run:
    python train_model.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("xgboost not installed - XGBoost will be skipped.")
    print("Install it with: pip install xgboost\n")


# =============================================================================
# Paths
# =============================================================================

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

MODELS_DIR.mkdir(exist_ok=True)
ALL_MODELS_DIR = MODELS_DIR / "all_models"
ALL_MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Selected features
# =============================================================================

DIABETES_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]


HEART_FEATURES = [
    "Sex",
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "PhysicalActivities",
    "SleepHours",
    "HadStroke",
    "HadAsthma",
    "HadKidneyDisease",
    "HadArthritis",
    "HadDiabetes",
    "SmokerStatus",
    "AgeCategory",
    "BMI",
    "AlcoholDrinkers",
]


KIDNEY_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "AlcoholConsumption",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "FamilyHistoryKidneyDisease",
    "FamilyHistoryHypertension",
    "FamilyHistoryDiabetes",
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
    "Edema",
]


# =============================================================================
# Settings
# =============================================================================

SVM_MAX_TRAIN_SAMPLES = 15000


# =============================================================================
# Helper functions
# =============================================================================

def clean_binary_target(series):
    """
    Convert common binary target formats to 0 and 1.
    """

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().unique())

        if unique_values.issubset({0, 1}):
            return series.astype(int)

    cleaned = series.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "disease": 1,
        "no disease": 0,
        "1": 1,
        "0": 0,
    }

    converted = cleaned.map(mapping)

    if converted.isna().any():
        unknown = cleaned[converted.isna()].unique()

        raise ValueError(
            f"Target contains values that could not be converted: {unknown}"
        )

    return converted.astype(int)


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def get_candidate_models(y_train):
    """
    Return candidate classifiers.
    """

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    scale_pos_weight = n_neg / max(n_pos, 1)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "SVM": SVC(
            probability=True,
            random_state=42,
            class_weight="balanced",
        ),

        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )

    return models


def subsample_for_svm(
    X_train,
    y_train,
    max_samples=SVM_MAX_TRAIN_SAMPLES,
):
    if len(X_train) <= max_samples:
        return X_train, y_train

    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_samples,
        stratify=y_train,
        random_state=42,
    )

    return X_sub, y_sub


def calculate_metrics(
    y_true,
    y_pred,
    y_proba=None,
):
    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
        labels=[0, 1],
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    f2_disease = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)),
            4,
        ),

        "precision_disease": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            4,
        ),

        "recall_disease": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            4,
        ),

        "f1_disease": round(
            float(f1_per_class[1]),
            4,
        ),

        "f1_no_disease": round(
            float(f1_per_class[0]),
            4,
        ),

        "macro_f1": round(
            float(macro_f1),
            4,
        ),

        "f2_disease": round(
            float(f2_disease),
            4,
        ),

        "roc_auc": None,
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(
            float(
                roc_auc_score(
                    y_true,
                    y_proba,
                )
            ),
            4,
        )

    return metrics


def find_best_threshold(
    y_true,
    y_proba,
    beta=2,
):
    """
    Find probability threshold that maximizes F-beta on validation data.
    """

    thresholds = np.arange(
        0.05,
        0.96,
        0.05,
    )

    rows = []

    for threshold in thresholds:
        y_pred = (
            y_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_true,
            y_pred,
            beta=beta,
            zero_division=0,
        )

        rows.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "precision": round(
                    float(precision),
                    4,
                ),
                "recall": round(
                    float(recall),
                    4,
                ),
                "f2": round(
                    float(f2),
                    4,
                ),
            }
        )

    best_row = max(
        rows,
        key=lambda row: row["f2"],
    )

    return (
        best_row["threshold"],
        rows,
        best_row,
    )


# =============================================================================
# Core training function
# =============================================================================

def train_and_benchmark(
    df,
    feature_columns,
    target_column,
    model_name,
    selection_metric="macro_f1",
    tune_threshold=False,
):

    print("\n" + "=" * 72)
    print(
        f"BENCHMARKING MODELS FOR: "
        f"{model_name.upper()}"
    )
    print("=" * 72)

    required_columns = (
        feature_columns
        + [target_column]
    )

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{model_name}: "
            f"Missing columns: "
            f"{missing_columns}"
        )

    df = df[
        required_columns
    ].copy()

    df = df.dropna(
        subset=[target_column]
    )

    df[target_column] = (
        clean_binary_target(
            df[target_column]
        )
    )

    X = df[feature_columns]
    y = df[target_column]

    numeric_features = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_features = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    print(
        f"Samples: {len(df):,}"
        f" | Positive: {(y == 1).sum():,}"
        f" | Negative: {(y == 0).sum():,}"
    )

    # -------------------------------------------------------------------------
    # Train / validation / test split
    #
    # 64% train
    # 16% validation
    # 20% test
    # -------------------------------------------------------------------------

    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=y_train_val,
    )

    print(
        f"Training rows   : "
        f"{len(X_train):,}"
    )

    print(
        f"Validation rows : "
        f"{len(X_val):,}"
    )

    print(
        f"Testing rows    : "
        f"{len(X_test):,}\n"
    )

    preprocessor = (
        build_preprocessor(
            numeric_features,
            categorical_features,
        )
    )

    candidate_models = (
        get_candidate_models(
            y_train
        )
    )

    results = []

    best_pipeline = None
    best_score = -1
    best_name = None

    print(
        f"{'Model':<20}"
        f"{'Accuracy':>10}"
        f"{'Disease F1':>12}"
        f"{'NoDis F1':>10}"
        f"{'Macro F1':>10}"
        f"{'F2':>10}"
        f"{'ROC-AUC':>10}"
    )

    print("-" * 82)

    print(
        f"Selecting winner by: "
        f"{selection_metric}\n"
    )

    # -------------------------------------------------------------------------
    # Benchmark on validation set
    # -------------------------------------------------------------------------

    for name, classifier in candidate_models.items():

        # Each candidate gets its own copy of the preprocessor: sklearn Pipelines
        # do not clone steps, so sharing one ColumnTransformer would let a later
        # fit (e.g. on the SVM subsample) silently overwrite an earlier model's.
        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(preprocessor),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        if name == "SVM":

            (
                X_fit,
                y_fit,
            ) = subsample_for_svm(
                X_train,
                y_train,
            )

            if len(X_fit) < len(X_train):
                print(
                    f"(SVM training "
                    f"subsample: "
                    f"{len(X_fit):,})"
                )

        else:
            X_fit = X_train
            y_fit = y_train

        pipeline.fit(
            X_fit,
            y_fit,
        )

        # Save every benchmark pipeline so the app can let users/doctors compare models.
        model_file_name = name.lower().replace(" ", "_")
        candidate_path = ALL_MODELS_DIR / f"{model_name}_{model_file_name}.pkl"
        with open(candidate_path, "wb") as candidate_file:
            pickle.dump(pipeline, candidate_file)

        y_pred_val = (
            pipeline.predict(
                X_val
            )
        )

        if hasattr(
            pipeline.named_steps[
                "classifier"
            ],
            "predict_proba",
        ):
            y_proba_val = (
                pipeline.predict_proba(
                    X_val
                )[:, 1]
            )
        else:
            y_proba_val = None

        metrics = calculate_metrics(
            y_val,
            y_pred_val,
            y_proba_val,
        )

        metrics["model"] = name

        results.append(metrics)

        print(
            f"{name:<20}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['f1_disease']:>12.4f}"
            f"{metrics['f1_no_disease']:>10.4f}"
            f"{metrics['macro_f1']:>10.4f}"
            f"{metrics['f2_disease']:>10.4f}"
            f"{(metrics['roc_auc'] or 0):>10.4f}"
        )

        score = metrics[
            selection_metric
        ]

        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = name

    print("-" * 82)

    print(
        f"Best validation model: "
        f"{best_name}"
    )

    print(
        f"{selection_metric}: "
        f"{best_score:.4f}"
    )

    # -------------------------------------------------------------------------
    # Threshold tuning on validation set
    # -------------------------------------------------------------------------

    decision_threshold = 0.5
    threshold_results = None

    if (
        tune_threshold
        and hasattr(
            best_pipeline.named_steps[
                "classifier"
            ],
            "predict_proba",
        )
    ):

        y_proba_val = (
            best_pipeline.predict_proba(
                X_val
            )[:, 1]
        )

        (
            best_threshold,
            threshold_rows,
            best_threshold_row,
        ) = find_best_threshold(
            y_val,
            y_proba_val,
            beta=2,
        )

        decision_threshold = (
            best_threshold
        )

        threshold_results = (
            threshold_rows
        )

        print(
            "\nHeart threshold tuning "
            "(validation set only)"
        )

        print(
            f"{'Threshold':>10}"
            f"{'Precision':>12}"
            f"{'Recall':>10}"
            f"{'F2':>10}"
        )

        for row in threshold_rows:

            marker = ""

            if (
                row["threshold"]
                == best_threshold
            ):
                marker = " <-- chosen"

            print(
                f"{row['threshold']:>10.2f}"
                f"{row['precision']:>12.4f}"
                f"{row['recall']:>10.4f}"
                f"{row['f2']:>10.4f}"
                f"{marker}"
            )

        print(
            f"\nChosen threshold: "
            f"{decision_threshold}"
        )

        print(
            f"Validation precision: "
            f"{best_threshold_row['precision']}"
        )

        print(
            f"Validation recall: "
            f"{best_threshold_row['recall']}"
        )

        print(
            f"Validation F2: "
            f"{best_threshold_row['f2']}"
        )

    # -------------------------------------------------------------------------
    # FINAL untouched test-set evaluation
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        f"FINAL TEST RESULTS: "
        f"{model_name.upper()}"
    )

    print("=" * 72)

    if (
        tune_threshold
        and decision_threshold != 0.5
        and hasattr(
            best_pipeline.named_steps[
                "classifier"
            ],
            "predict_proba",
        )
    ):

        y_proba_test = (
            best_pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        y_pred_test = (
            y_proba_test
            >= decision_threshold
        ).astype(int)

    else:

        y_pred_test = (
            best_pipeline.predict(
                X_test
            )
        )

        if hasattr(
            best_pipeline.named_steps[
                "classifier"
            ],
            "predict_proba",
        ):
            y_proba_test = (
                best_pipeline.predict_proba(
                    X_test
                )[:, 1]
            )
        else:
            y_proba_test = None

    final_metrics = calculate_metrics(
        y_test,
        y_pred_test,
        y_proba_test,
    )

    print(
        f"\nBest model: "
        f"{best_name}"
    )

    print(
        f"Decision threshold: "
        f"{decision_threshold}"
    )

    print(
        f"\nAccuracy: "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Disease precision: "
        f"{final_metrics['precision_disease']:.4f}"
    )

    print(
        f"Disease recall: "
        f"{final_metrics['recall_disease']:.4f}"
    )

    print(
        f"Disease F1: "
        f"{final_metrics['f1_disease']:.4f}"
    )

    print(
        f"No Disease F1: "
        f"{final_metrics['f1_no_disease']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{final_metrics['macro_f1']:.4f}"
    )

    print(
        f"Disease F2: "
        f"{final_metrics['f2_disease']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{final_metrics['roc_auc']}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred_test,
            target_names=[
                "No Disease",
                "Disease",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred_test,
        )
    )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    model_path = (
        MODELS_DIR
        / f"{model_name}_model.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as file:

        pickle.dump(
            best_pipeline,
            file,
        )

    print(
        f"\nSaved model -> "
        f"{model_path}"
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    return {
        "features": feature_columns,
        "target": target_column,

        "n_samples": len(df),

        "positive_cases": int(
            (y == 1).sum()
        ),

        "negative_cases": int(
            (y == 0).sum()
        ),

        "numeric_features": (
            numeric_features
        ),

        "categorical_features": (
            categorical_features
        ),

        "best_model": best_name,

        "selection_metric": (
            selection_metric
        ),

        "best_validation_score": (
            round(
                float(best_score),
                4,
            )
        ),

        "decision_threshold": (
            decision_threshold
        ),

        "validation_model_results": (
            results
        ),

        "threshold_results": (
            threshold_results
        ),

        "final_test_metrics": (
            final_metrics
        ),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    metadata = {}

    disease_configs = [
        (
            "diabetes",
            "diabetes.csv",
            DIABETES_FEATURES,
            "diabetes",
            "macro_f1",
            False,
        ),

        (
            "heart",
            "heart.csv",
            HEART_FEATURES,
            "HadHeartAttack",
            "f2_disease",
            True,
        ),

        (
            "kidney",
            "kidney_disease.csv",
            KIDNEY_FEATURES,
            "Diagnosis",
            "macro_f1",
            False,
        ),
    ]

    for (
        model_name,
        csv_name,
        feature_list,
        target_col,
        selection_metric,
        tune_threshold,
    ) in disease_configs:

        print(
            f"\nLoading "
            f"{model_name} dataset..."
        )

        csv_path = (
            DATA_DIR
            / csv_name
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: "
                f"{csv_path}"
            )

        df = pd.read_csv(
            csv_path
        )

        metadata[
            model_name
        ] = train_and_benchmark(
            df=df,
            feature_columns=feature_list,
            target_column=target_col,
            model_name=model_name,
            selection_metric=selection_metric,
            tune_threshold=tune_threshold,
        )

    # =========================================================================
    # Save metadata
    # =========================================================================

    metadata_path = (
        MODELS_DIR
        / "model_meta.json"
    )

    with open(
        metadata_path,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nMetadata saved -> "
        f"{metadata_path}"
    )

    # =========================================================================
    # Final summary
    # =========================================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "ALL DISEASES "
        "BENCHMARKED SUCCESSFULLY"
    )

    print("=" * 72)

    for (
        model_name,
        _,
        _,
        _,
        _,
        _,
    ) in disease_configs:

        info = metadata[
            model_name
        ]

        print(
            f"{model_name.capitalize():<10}"
            f" best model: "
            f"{info['best_model']:<20}"
            f" validation "
            f"{info['selection_metric']}="
            f"{info['best_validation_score']:.4f}"
            f" threshold="
            f"{info['decision_threshold']}"
        )

    print("=" * 72)