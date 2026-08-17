import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 10 — ANNUAL TREND ANALYSIS
# ============================================================
#
# Purpose:
# Determine whether burden for each risk factor is:
# Increasing / Decreasing / Stable
#
# Formula:
#
# YoY % =
# (Current Year - Previous Year)
# / Previous Year * 100
#
#
# Usage:
#
# python annual_trend_analysis.py .\normalise_gbd_dataset\
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

MEASURE = "DALYs"

# IMPORTANT:
# Change this to "Rate" if Section 10 should use Rate.
METRIC = "Number"

# Fixed age band.
# Do NOT use "India" or "Both" because your files contain
# state-level locations and Male/Female.
AGE = "35-39 years"

START_YEAR = 2013
END_YEAR = 2023

# Output directory
OUTPUT_DIR = Path("section_10_output")


# ============================================================
# TREND CLASSIFICATION
# ============================================================

# Threshold for classifying yearly average change.
#
# Example:
# +0.5% to -0.5% = Stable
#
STABLE_THRESHOLD = 0.5


# ============================================================
# FIND DALYs FILES
# ============================================================

def find_dalys_files(root):

    files = []

    for file in root.rglob("*.csv"):

        parts = [
            p.lower()
            for p in file.relative_to(root).parts
        ]

        if "dalys" in parts:
            files.append(file)

    return files


# ============================================================
# READ FILE
# ============================================================

def read_file(file):

    try:

        df = pd.read_csv(file)

    except Exception as e:

        print()
        print(
            f"[ERROR] Could not read:"
        )

        print(file)
        print(e)

        return None

    return df


# ============================================================
# FILTER FILE
# ============================================================

def filter_file(df, file):

    required_columns = [
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "rei_id",
        "rei_name",
        "year",
        "val",
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        print()
        print(
            f"[SKIP] Missing columns in:"
        )

        print(file)

        print(
            "Missing:",
            missing
        )

        return None

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    if "metric_name" not in df.columns:

        print()
        print(
            f"[SKIP] metric_name column missing:"
        )

        print(file)

        return None

    df["metric_name"] = (
        df["metric_name"]
        .astype(str)
        .str.strip()
    )

    df["age_name"] = (
        df["age_name"]
        .astype(str)
        .str.strip()
    )

    df["sex_name"] = (
        df["sex_name"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Filter metric
    # --------------------------------------------------------

    df = df[
        df["metric_name"].str.lower()
        == METRIC.lower()
    ]

    # --------------------------------------------------------
    # Filter age
    # --------------------------------------------------------

    df = df[
        df["age_name"].str.lower()
        == AGE.lower()
    ]

    # --------------------------------------------------------
    # Years
    # --------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df = df[
        df["year"].between(
            START_YEAR,
            END_YEAR
        )
    ]

    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------
    #
    # We use Male + Female.
    # Do NOT expect "Both".
    #

    df = df[
        df["sex_name"].isin(
            [
                "Male",
                "Female",
            ]
        )
    ]

    return df


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_data(root):

    files = find_dalys_files(
        root
    )

    print("=" * 80)
    print("SECTION 10 — ANNUAL TREND ANALYSIS")
    print("=" * 80)

    print()

    print(
        f"Root folder : {root.resolve()}"
    )

    print(
        f"DALYs files found : {len(files)}"
    )

    print(
        f"Measure : {MEASURE}"
    )

    print(
        f"Metric  : {METRIC}"
    )

    print(
        f"Age     : {AGE}"
    )

    print(
        f"Years   : {START_YEAR}-{END_YEAR}"
    )

    print()

    all_data = []

    for index, file in enumerate(
        files,
        start=1
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{file.name}"
        )

        df = read_file(file)

        if df is None:
            continue

        df = filter_file(
            df,
            file
        )

        if df is None:
            continue

        if df.empty:
            continue

        # Add source information
        df["source_file"] = str(
            file.relative_to(root)
        )

        all_data.append(df)

    if not all_data:

        print()
        print(
            "NO DATA FOUND AFTER FILTERING."
        )

        print()
        print(
            "Check:"
        )

        print(
            f"Metric = {METRIC}"
        )

        print(
            f"Age = {AGE}"
        )

        sys.exit(1)

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    return combined


# ============================================================
# REMOVE DUPLICATE ROWS
# ============================================================

def remove_duplicates(df):

    before = len(df)

    key_columns = [
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "rei_id",
        "rei_name",
        "year",
    ]

    existing_keys = [
        c
        for c in key_columns
        if c in df.columns
    ]

    df = df.drop_duplicates(
        subset=existing_keys
    )

    after = len(df)

    print()
    print(
        "Duplicate handling:"
    )

    print(
        f"Rows before : {before:,}"
    )

    print(
        f"Rows after  : {after:,}"
    )

    print(
        f"Duplicates removed : "
        f"{before - after:,}"
    )

    return df


# ============================================================
# CALCULATE YEARLY BURDEN
# ============================================================

def calculate_yearly_burden(df):

    print()
    print("=" * 80)
    print("CALCULATING YEARLY BURDEN")
    print("=" * 80)

    # --------------------------------------------------------
    # Convert val to numeric
    # --------------------------------------------------------

    df["val"] = pd.to_numeric(
        df["val"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["val"]
    )

    # --------------------------------------------------------
    # Group
    # --------------------------------------------------------
    #
    # Risk factor + year
    #
    # This sums:
    #   - all states
    #   - Male + Female
    #   - all causes present
    #
    # Age remains fixed.
    #

    yearly = (
        df.groupby(
            [
                "rei_id",
                "rei_name",
                "year",
            ],
            as_index=False
        )["val"]
        .sum()
    )

    yearly = yearly.sort_values(
        [
            "rei_name",
            "year",
        ]
    )

    return yearly


# ============================================================
# YOY CALCULATION
# ============================================================

def calculate_yoy(yearly):

    print()
    print("=" * 80)
    print("CALCULATING YOY % CHANGE")
    print("=" * 80)

    yearly = yearly.copy()

    yearly["previous_val"] = (
        yearly
        .groupby(
            [
                "rei_id",
                "rei_name",
            ]
        )["val"]
        .shift(1)
    )

    # --------------------------------------------------------
    # YoY
    # --------------------------------------------------------

    yearly["yoy_pct_change"] = (
        (
            yearly["val"]
            - yearly["previous_val"]
        )
        / yearly["previous_val"]
        * 100
    )

    # --------------------------------------------------------
    # First year has no previous year
    # --------------------------------------------------------

    yearly.loc[
        yearly["previous_val"].isna(),
        "yoy_pct_change"
    ] = pd.NA

    # --------------------------------------------------------
    # Previous value = 0
    # --------------------------------------------------------

    yearly.loc[
        yearly["previous_val"] == 0,
        "yoy_pct_change"
    ] = pd.NA

    return yearly


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trends(yearly):

    print()
    print("=" * 80)
    print("TREND CLASSIFICATION")
    print("=" * 80)

    results = []

    for (rei_id, rei_name), group in (
        yearly.groupby(
            [
                "rei_id",
                "rei_name",
            ]
        )
    ):

        group = group.sort_values(
            "year"
        )

        yoy = group[
            "yoy_pct_change"
        ].dropna()

        if yoy.empty:

            direction = "Insufficient data"

            avg_yoy = pd.NA

        else:

            avg_yoy = yoy.mean()

            if avg_yoy > STABLE_THRESHOLD:

                direction = "Increasing"

            elif avg_yoy < -STABLE_THRESHOLD:

                direction = "Decreasing"

            else:

                direction = "Stable"

        results.append(
            {
                "rei_id": rei_id,
                "rei_name": rei_name,
                "average_yoy_pct": avg_yoy,
                "trend_direction": direction,
                "years_available": group[
                    "year"
                ].nunique(),
            }
        )

    trend_df = pd.DataFrame(
        results
    )

    return trend_df


# ============================================================
# LINE GRAPHS
# ============================================================

def create_graphs(yearly):

    graph_dir = (
        OUTPUT_DIR
        / "trend_graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 80)
    print("CREATING LINE GRAPHS")
    print("=" * 80)

    for rei_id, group in yearly.groupby(
        "rei_id"
    ):

        group = group.sort_values(
            "year"
        )

        rei_name = (
            group["rei_name"]
            .iloc[0]
        )

        plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            group["year"],
            group["val"],
            marker="o"
        )

        plt.title(
            f"Annual Trend — {rei_name}"
        )

        plt.xlabel(
            "Year"
        )

        plt.ylabel(
            f"{MEASURE} ({METRIC})"
        )

        plt.xticks(
            range(
                START_YEAR,
                END_YEAR + 1
            )
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        safe_name = "".join(
            c if c.isalnum() or c in " _-"
            else "_"
            for c in rei_name
        )

        output_file = (
            graph_dir
            / f"{rei_id}_{safe_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=150
        )

        plt.close()

    print(
        f"Graphs saved to:"
    )

    print(
        graph_dir.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python annual_trend_analysis.py "
            "<normalized_folder>"
        )

        print()

        print(
            "Example:"
        )


        sys.exit(1)

    root = Path(
        sys.argv[1]
    )

    if not root.exists():

        print(
            f"Folder not found:"
        )

        print(
            root.resolve()
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        root
    )

    print()
    print(
        f"Rows loaded after filtering: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Yearly burden
    # --------------------------------------------------------

    yearly = calculate_yearly_burden(
        df
    )

    # --------------------------------------------------------
    # YoY
    # --------------------------------------------------------

    yearly = calculate_yoy(
        yearly
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = classify_trends(
        yearly
    )

    # --------------------------------------------------------
    # Save yearly table
    # --------------------------------------------------------

    yearly_file = (
        OUTPUT_DIR
        / "annual_trend_yoy.csv"
    )

    yearly.to_csv(
        yearly_file,
        index=False
    )

    # --------------------------------------------------------
    # Save trend summary
    # --------------------------------------------------------

    trend_file = (
        OUTPUT_DIR
        / "annual_trend_summary.csv"
    )

    trend.to_csv(
        trend_file,
        index=False
    )

    # --------------------------------------------------------
    # Create graphs
    # --------------------------------------------------------

    create_graphs(
        yearly
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("SECTION 10 COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Risk factors analysed : "
        f"{yearly['rei_id'].nunique()}"
    )

    print(
        f"Years : "
        f"{yearly['year'].min()}-"
        f"{yearly['year'].max()}"
    )

    print()

    print(
        "Trend classification:"
    )

    print(
        trend[
            [
                "rei_id",
                "rei_name",
                "average_yoy_pct",
                "trend_direction",
                "years_available",
            ]
        ]
        .sort_values(
            "rei_name"
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Yearly YoY table:"
    )

    print(
        yearly[
            [
                "rei_id",
                "rei_name",
                "year",
                "val",
                "previous_val",
                "yoy_pct_change",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Output files:"
    )

    print(
        yearly_file.resolve()
    )

    print(
        trend_file.resolve()
    )


if __name__ == "__main__":
    main()