"""
Run this ONCE after training to extract real historical medians from your CSV
and print the exact dict literals to paste into app.py.

Usage:
    python extract_medians.py --data data.csv
"""

import argparse
import json
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to training CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data, na_values=["NULL", "Null", "null", "", "NaN"])

    ts_fmt = "%Y-%m-%d %H:%M:%S.%f%z"
    for col in ["start_datetime", "closed_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=ts_fmt, errors="coerce")

    df = df.dropna(subset=["start_datetime", "closed_datetime"])
    df["resolution_time_mins"] = (
        df["closed_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 60.0
    df = df[df["resolution_time_mins"] > 0]

    global_median = float(df["resolution_time_mins"].median())

    zone_medians = (
        df.groupby("zone")["resolution_time_mins"]
        .median()
        .round(2)
        .to_dict()
    )
    zone_medians = {str(k): v for k, v in zone_medians.items()}

    junction_medians = {}
    if "junction" in df.columns:
        junction_medians = (
            df.groupby("junction")["resolution_time_mins"]
            .median()
            .round(2)
            .to_dict()
        )
        junction_medians = {str(k): v for k, v in junction_medians.items()}

    print("\n# ── Paste these into app.py ──────────────────────────────────")
    print(f"GLOBAL_MEDIAN = {global_median}")
    print(f"\nZONE_MEDIANS: dict[str, float] = {json.dumps(zone_medians, indent=4)}")
    if junction_medians:
        print(f"\nJUNCTION_MEDIANS: dict[str, float] = {json.dumps(junction_medians, indent=4)}")
    print("# ────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
