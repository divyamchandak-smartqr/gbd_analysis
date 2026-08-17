import sys
from pathlib import Path
import pandas as pd


# ============================================================
# SECTION 10 — ANNUAL TREND ANALYSIS
# ============================================================
#
# Formula:
#
# Annual Change (%) =
#     (Current Year Value - Prior Year Value)
#     / Prior Year Value * 100
#
# Usage:
#
# python annual_trend_analysis.py <normalized_folder>
#
# Example:
#
# python annual_trend_analysis.py .\normalise_gbd_dataset\
#
# ============================================================


MEASURE = "DALYs"
METRIC = "Number"

START_YEAR = 2013
END_YEAR = 2023

# ------------------------------------------------------------
# Set these after checking the available values printed by
# the script.
# ------------------------------------------------------------

LOCATION = "India"
SEX = "Both"

# Keep None initially so the script can show available ages.
AGE = None


OUTPUT_DIR = Path("section_10_output")


# ============================================================
# FIND MEASURE
# ============================================================

def get_measure(csv_file, root):

    parts = csv_file.relative_to(root).parts

    for part in parts:

        p = part.strip().lower()

        if p == "deaths":
            return "Deaths"

        if p == "dalys":
            return "DALYs"

        if p == "ylls":
            return "YLLs"

        if p == "ylds":
            return "YLDs"

    return None


# ============================================================
# FIND METRIC FROM FILE CONTENT
# ============================================================

def read_csv(csv_file):

    try:
        return pd.read_csv(csv_file)

    except Exception as e:

        print(f"[ERROR] {csv_file}")
        print(f"        {e}")

        return None


# ============================================================
# SCAN FILES
# ============================================================

def scan_dataset(root):

    csv_files = list(
        root.rglob("*.csv")
    )

    print("=" * 80)
    print("SECTION 10 — ANNUAL TREND ANALYSIS")
    print("=" * 80)

    print(
        f"Root folder : {root.resolve()}"
    )

    print(
        f"CSV files   : {len(csv_files)}"
    )

    selected_files = []

    for csv_file in csv_files:

        measure = get_measure(
            csv_file,
            root
        )

        if measure != MEASURE:
            continue

        selected_files.append(
            csv_file
        )

    print(
        f"{MEASURE} files: "
        f"{len(selected_files)}"
    )

    return selected_files


# ============================================================
# LOAD DATA
# ============================================================

def load_data(files):

    required_columns = [
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "rei_name",
        "year",
        "val",
        "upper",
        "lower",
    ]

    frames = []

    for csv_file in files:

        try:

            df = pd.read_csv(
                csv_file
            )

        except Exception as e:

            print(
                f"[ERROR] Reading {csv_file}"
            )

            print(e)

            continue

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            print(
                f"[SKIP] Missing columns "
                f"{missing}: {csv_file}"
            )

            continue

        # ----------------------------------------------------
        # Metric check
        # ----------------------------------------------------

        if "metric_name" in df.columns:

            metric_values = (
                df["metric_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )

            if METRIC not in metric_values:

                continue

            df = df[
                df["metric_name"]
                .astype(str)
                .str.strip()
                == METRIC
            ]

        else:

            # Metric is encoded in filename.
            filename = csv_file.name.lower()

            if METRIC.lower() not in filename:

                continue

        frames.append(df)

    if not frames:

        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# SHOW AVAILABLE FILTER VALUES
# ============================================================

def show_available_values(df):

    print()
    print("=" * 80)
    print("AVAILABLE VALUES")
    print("=" * 80)

    print()
    print("Locations:")
    print(
        sorted(
            df["location_name"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    print()
    print("Sex:")
    print(
        sorted(
            df["sex_name"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    print()
    print("Age:")
    print(
        sorted(
            df["age_name"]
            .dropna()
            .astype(str)
            .unique()
        )
    )

    print()
    print("Years:")
    print(
        sorted(
            df["year"]
            .dropna()
            .unique()
        )
    )


# ============================================================
# FILTER DATA
# ============================================================

def filter_data(df):

    result = df.copy()

    result = result[
        result["year"].between(
            START_YEAR,
            END_YEAR
        )
    ]

    result = result[
        result["location_name"]
        .astype(str)
        .str.strip()
        == LOCATION
    ]

    result = result[
        result["sex_name"]
        .astype(str)
        .str.strip()
        == SEX
    ]

    if AGE is not None:

        result = result[
            result["age_name"]
            .astype(str)
            .str.strip()
            == AGE
        ]

    return result


# ============================================================
# CALCULATE ANNUAL TREND
# ============================================================

def calculate_trend(df):

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Group by risk factor + year
    # --------------------------------------------------------

    yearly = (
        df.groupby(
            [
                "rei_id",
                "rei_name",
                "year"
            ],
            as_index=False
        )["val"]
        .sum()
    )

    yearly = yearly.sort_values(
        [
            "rei_id",
            "year"
        ]
    )

    # --------------------------------------------------------
    # Previous year value
    # --------------------------------------------------------

    yearly["prior_year_val"] = (
        yearly
        .groupby("rei_id")["val"]
        .shift(1)
    )

    # --------------------------------------------------------
    # YoY percentage change
    # --------------------------------------------------------

    yearly["yoy_pct_change"] = (
        (
            yearly["val"]
            - yearly["prior_year_val"]
        )
        / yearly["prior_year_val"]
        * 100
    )

    # --------------------------------------------------------
    # First year has no prior year
    # --------------------------------------------------------

    yearly.loc[
        yearly["prior_year_val"].isna(),
        "yoy_pct_change"
    ] = None

    return yearly


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Command line
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python annual_trend_analysis.py "
            "<normalized_folder>"
        )

        print()
        print("Example:")

        print(
            r"python annual_trend_analysis.py "
        )

        sys.exit(1)

    root = Path(
        sys.argv[1]
    )

    if not root.exists():

        print(
            f"Folder not found: {root}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    files = scan_dataset(
        root
    )

    if not files:

        print()
        print(
            f"No {MEASURE} files found."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        files
    )

    if df.empty:

        print()
        print(
            "No usable data found."
        )

        sys.exit(1)

    print()
    print(
        f"Rows loaded: {len(df):,}"
    )

    # --------------------------------------------------------
    # Show available values
    # --------------------------------------------------------

    show_available_values(
        df
    )

    # --------------------------------------------------------
    # Filter
    # --------------------------------------------------------

    filtered = filter_data(
        df
    )

    print()
    print("=" * 80)
    print("FILTER")
    print("=" * 80)

    print(
        f"Measure : {MEASURE}"
    )

    print(
        f"Metric  : {METRIC}"
    )

    print(
        f"Location: {LOCATION}"
    )

    print(
        f"Sex     : {SEX}"
    )

    print(
        f"Age     : {AGE}"
    )

    print(
        f"Years   : {START_YEAR}-{END_YEAR}"
    )

    print(
        f"Rows after filtering: "
        f"{len(filtered):,}"
    )

    if filtered.empty:

        print()
        print(
            "NO DATA AFTER FILTERING."
        )

        print(
            "Check LOCATION, SEX and AGE "
            "against the available values printed above."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = calculate_trend(
        filtered
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "annual_trend_dalys.csv"
    )

    trend.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("ANNUAL TREND RESULT")
    print("=" * 80)

    print(
        trend.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Output:\n{output_file.resolve()}"
    )


if __name__ == "__main__":
    main()