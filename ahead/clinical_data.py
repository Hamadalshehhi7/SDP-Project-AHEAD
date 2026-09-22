"""Validate uploaded records before they can reach the screening pipeline."""

import math
import re
from datetime import date, datetime

import pandas as pd

from ahead.config import GENDER_CODED_FIELDS


def age_from_dob(value, today=None):
    if value is None or str(value).strip() == "":
        return None
    day = date.fromisoformat(str(value)[:10])
    today = today or date.today()
    years = today.year - day.year - ((today.month, today.day) < (day.month, day.day))
    if not 0 <= years <= 120:
        raise ValueError("Date of birth must describe an age from 0 to 120.")
    return years


def age_category(age, choices):
    for choice in choices:
        numbers = [int(n) for n in re.findall(r"\d+", str(choice))]
        if len(numbers) == 2 and numbers[0] <= age <= numbers[1]:
            return choice
        if len(numbers) == 1 and age >= numbers[0] and ("older" in str(choice).lower() or "+" in str(choice)):
            return choice
    raise ValueError("Date of birth does not match the model's age categories.")


def validate_row(values, disease, required, stats, dob=None):
    """Return normalized model inputs and actionable issues; never silently score an invalid row."""
    clean, issues = {}, []
    age_key = "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    try:
        age = age_from_dob(dob)
        if age is not None:
            values[age_key] = age_category(age, stats[age_key]["options"]) if disease == "heart" else age
    except (ValueError, TypeError):
        issues.append("Invalid date of birth; use YYYY-MM-DD.")
    for feature in required:
        value = values.get(feature)
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            issues.append(f"{feature}: missing; enter a value before scoring.")
            clean[feature] = None
            continue
        info = stats[feature]
        if info["kind"] == "number":
            if isinstance(value, str) and info.get("binary"):
                mapping = {"yes": 1, "true": 1, "y": 1, "no": 0, "false": 0, "n": 0}
                if feature in GENDER_CODED_FIELDS:
                    mapping.update({"male": 0, "m": 0, "female": 1, "f": 1})
                value = mapping.get(value.strip().lower(), value)
            try:
                number = float(value)
            except (ValueError, TypeError):
                number = float("nan")
            if not math.isfinite(number):
                issues.append(f"{feature}: enter a number.")
            elif info.get("integer") and not number.is_integer():
                issues.append(f"{feature}: enter a whole number.")
            elif info.get("binary") and number not in (0, 1):
                issues.append(f"{feature}: enter 0 or 1.")
            elif number < info["min"] or number > info["max"]:
                issues.append(f"{feature}: {number:g} is outside the training range ({info['min']:g}–{info['max']:g}); verify it.")
            clean[feature] = (int(number) if info["integer"] else number) if math.isfinite(number) else None
        else:
            options = {str(option).strip().lower(): option for option in info["options"]}
            matched = options.get(str(value).strip().lower())
            if matched is None:
                issues.append(f"{feature}: choose a category from the training data.")
            clean[feature] = matched
    return clean, issues


def prepare_upload(frame, disease, required, stats, id_columns):
    """Read rows without requiring age at upload; all other model columns must exist."""
    age_key = "AgeCategory" if disease == "heart" else "Age" if disease == "kidney" else "age"
    absent = [f for f in required if f not in frame.columns and f != age_key]
    if absent:
        raise ValueError("Missing model columns: " + ", ".join(absent))
    id_column = next((c for c in id_columns if c in frame.columns), None)
    dob_column = next((c for c in ("DOB", "dob", "DateOfBirth", "date_of_birth", "Date of Birth") if c in frame.columns), None)
    output = []
    for index, (_, row) in enumerate(frame.iterrows(), 1):
        dob = row[dob_column] if dob_column else None
        dob = None if dob is None or pd.isna(dob) else str(dob)[:10]
        values = {f: row[f] if f in frame.columns else None for f in required}
        values = {f: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for f, v in values.items()}
        clean, issues = validate_row(values, disease, required, stats, dob)
        patient = str(row[id_column]) if id_column and pd.notna(row[id_column]) else f"Row {index}"
        output.append({"patient_id": patient[:100], "values": clean, "dob": dob, "issues": issues})
    return output
