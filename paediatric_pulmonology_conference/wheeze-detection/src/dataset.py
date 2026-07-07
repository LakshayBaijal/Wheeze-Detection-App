"""
Step 1 - Read the dataset and turn it into a tidy table of labelled events.

SPRSound stores, for each recording:
  * a .wav audio file   (e.g.  65101170_0.4_0_p1_3246.wav)
  * a .json file with the same name, listing each breathing "event" inside the
    recording, with its start time, end time, and a label (Normal, Wheeze, ...).

The first part of the filename (65101170) is the PATIENT number. We keep track
of it so that, when we split the data, no single child appears in both the
training set and the test set (that would let the model "cheat").

This file produces one row per event:
    wav_path | start_ms | end_ms | patient_id | label (wheeze / no_wheeze)
"""

import json

import pandas as pd

from config import (
    JSON_SUBDIR,
    WAV_SUBDIR,
    WHEEZE_EVENT_LABELS,
)


def _patient_id_from_filename(wav_name: str) -> str:
    """The patient number is the first underscore-separated part of the name."""
    return wav_name.split("_")[0]


def _read_events_from_json(json_path):
    """
    Return a list of (start_ms, end_ms, type_string) for every event in one file.

    SPRSound JSON files have varied slightly between challenge years, so we look
    for the event list under a few possible key names and stay tolerant.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The list of events may be stored under one of these keys.
    events = None
    for key in ("event_annotation", "events", "event", "annotation"):
        if isinstance(data.get(key), list):
            events = data[key]
            break
    if events is None:
        return []

    parsed = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        # Each event has start / end (milliseconds, as text) and a type label.
        start = ev.get("start", ev.get("start_time"))
        end = ev.get("end", ev.get("end_time"))
        etype = ev.get("type", ev.get("label"))
        if start is None or end is None or etype is None:
            continue
        try:
            parsed.append((int(float(start)), int(float(end)), str(etype).strip()))
        except (TypeError, ValueError):
            continue
    return parsed


def build_event_table(wav_dir=WAV_SUBDIR, json_dir=JSON_SUBDIR) -> pd.DataFrame:
    """Scan every recording and build the full table of labelled events."""
    rows = []
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {wav_dir}. "
            "Did you run  python src/download_data.py  first, and is the path "
            "in src/config.py correct?"
        )

    for wav_path in wav_files:
        json_path = json_dir / (wav_path.stem + ".json")
        if not json_path.exists():
            continue

        patient_id = _patient_id_from_filename(wav_path.name)
        for start_ms, end_ms, etype in _read_events_from_json(json_path):
            if end_ms <= start_ms:
                continue
            label = "wheeze" if etype in WHEEZE_EVENT_LABELS else "no_wheeze"
            rows.append(
                {
                    "wav_path": str(wav_path),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "patient_id": patient_id,
                    "event_type": etype,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "Parsed 0 events. The JSON layout may differ from what this script "
            "expects - open one .json file in data/SPRSound and check the keys, "
            "then adjust _read_events_from_json() in src/dataset.py."
        )
    return df


def split_by_patient(df: pd.DataFrame, test_fraction, val_fraction, seed):
    """
    Split into train / val / test so that all events from one child stay in the
    SAME split. We shuffle the list of patients (reproducibly) and slice it.
    """
    patients = sorted(df["patient_id"].unique())
    rng = pd.Series(patients).sample(frac=1.0, random_state=seed).tolist()

    n = len(rng)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))

    test_p = set(rng[:n_test])
    val_p = set(rng[n_test : n_test + n_val])
    train_p = set(rng[n_test + n_val :])

    def tag(pid):
        if pid in test_p:
            return "test"
        if pid in val_p:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df["patient_id"].map(tag)
    return df


def summarise(df: pd.DataFrame):
    """Print a quick, human-readable summary of the data."""
    print("\n=== DATA SUMMARY ===")
    print(f"Total events : {len(df):,}")
    print(f"Patients     : {df['patient_id'].nunique():,}")
    print("\nEvents per label:")
    print(df["label"].value_counts().to_string())
    if "split" in df.columns:
        print("\nEvents per split:")
        print(df.groupby(["split", "label"]).size().to_string())
    print("====================\n")


if __name__ == "__main__":
    # Quick manual test: build the table and show a summary.
    from config import RANDOM_SEED, TEST_FRACTION, VAL_FRACTION

    table = build_event_table()
    table = split_by_patient(table, TEST_FRACTION, VAL_FRACTION, RANDOM_SEED)
    summarise(table)
    print(table.head(10).to_string())
