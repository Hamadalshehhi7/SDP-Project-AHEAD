"""
utils/helpers.py
================
Shared helper functions used across the app.
"""

import pandas as pd
import numpy as np


def bmi_category(bmi: float) -> str:
    """Return a human-readable BMI category."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def glucose_category(glucose: float) -> str:
    """Classify glucose level."""
    if glucose < 70:
        return "Low (Hypoglycaemia)"
    elif glucose < 100:
        return "Normal"
    elif glucose < 126:
        return "Pre-diabetic"
    else:
        return "Diabetic range"


def blood_pressure_category(bp: float) -> str:
    """Classify diastolic blood pressure."""
    if bp < 60:
        return "Low"
    elif bp < 80:
        return "Normal"
    elif bp < 90:
        return "Elevated"
    else:
        return "High"


def risk_level(probability: float) -> tuple[str, str]:
    """Return (label, colour) based on probability."""
    if probability < 0.30:
        return "Low", "#34D399"
    elif probability < 0.60:
        return "Moderate", "#FBBF24"
    else:
        return "High", "#F87171"


def compute_stats(df: pd.DataFrame, target_col: str) -> dict:
    """Compute basic descriptive stats for the dataset."""
    pos = (df[target_col] == 1).sum()
    neg = (df[target_col] == 0).sum()
    return {
        "total": len(df),
        "positive": int(pos),
        "negative": int(neg),
        "positive_pct": round(pos / len(df) * 100, 1),
    }
