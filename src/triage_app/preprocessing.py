"""Load and clean the emergency triage dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "AGE",
    "SEX",
    "TEMPF",
    "PULSE",
    "RESPR",
    "BPSYS",
    "BPDIAS",
    "POPCT",
    "PAINSCALE",
    "ARREMS",
    "IMMEDR",
]

FEATURE_COLUMNS = REQUIRED_COLUMNS[:-1]

BINARY_TRIAGE_LABELS = {
    1: 1,  # Immediate
    2: 1,  # Emergent
    3: 1,  # Urgent
    4: 0,  # Semi-Urgent
    5: 0,  # Non-Urgent
}
TRIAGE_NAMES = {0: "Non-Urgent", 1: "Urgent"}
TRIAGE_ORDER = list(TRIAGE_NAMES)


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Read the source CSV and return cleaned model-ready columns."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Place "
            "'Emergency_Triage_Real_Dataset.csv' beside main.py."
        )

    raw_data = pd.read_csv(csv_path)
    raw_data.columns = raw_data.columns.str.upper()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw_data]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = raw_data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace([-9, -8], np.nan)
    data["PULSE"] = data["PULSE"].replace(998, np.nan)
    data["BPDIAS"] = data["BPDIAS"].replace(998, np.nan)

    # The source records temperature with an implied decimal place.
    data["TEMPF"] = data["TEMPF"] / 10

    data = data[data["IMMEDR"].isin(BINARY_TRIAGE_LABELS)].copy()
    data["IMMEDR"] = data["IMMEDR"].map(BINARY_TRIAGE_LABELS).astype(int)
    data["SEX"] = data["SEX"].map({1: "Female", 2: "Male"})
    data["ARREMS"] = data["ARREMS"].map({1: "Yes", 2: "No"})

    if data.empty:
        raise ValueError("No rows with valid triage categories remain after cleaning.")

    return data
