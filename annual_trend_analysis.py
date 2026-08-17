import sys
from pathlib import Path
import pandas as pd


# ============================================================
# SECTION 10 — ANNUAL TREND ANALYSIS
# ============================================================

MEASURE = "DALYs"
METRIC = "Number"

START_YEAR = 2013
END_YEAR = 2023

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# Your GBD export has:
#
#   31 locations
#   Male
#   Female
#   10 age bands
#
# There is NO:
#   India
#   Both
#
# Therefore we aggregate all states/UTs and both sexes.
#
# Set ONE age band here.
# ------------------------------------------------------------

AGE = "20-24 years"

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
    ]

    frames = []

    for csv_file in files:

        try:

            df = pd.read_csv(
                csv_file
            )

        except Exception as e:

            print(
                f"[ERROR] {csv_file}"
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
        # Metric
        # ----------------------------------------------------

        if "metric_name" in df.columns:

            df["metric_name"] = (
                df["metric_name"]
                .astype(str)
                .str.strip()
            )

            df = df[
                df["metric_name"] == METRIC
            ]

        else:

            # Fallback: metric encoded in filename
            if METRIC.lower() not in (
                csv_file.name.lower()
            ):
                continue

        frames.append(
            df[
                required_columns
            ]
        )

    if not frames:

        return pd.DataFrame(
            columns=required_columns
        )

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# CALCULATE TREND
# ============================================================

def calculate_trend(df):

    # --------------------------------------------------------
    # Filter years and age
    # --------------------------------------------------------

    df = df[
        df["year"].between(
            START_YEAR,
            END_YEAR
        )
    ]

    df = df[
        df["age_name"]
        .astype(str)
        .str.strip()
        == AGE
    ]

    # --------------------------------------------------------
    # We intentionally DO NOT filter location.
    #
    # All 31 states/UTs are included.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # We intentionally DO NOT filter sex.
    #
    # Male + Female are included.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Sum across:
    #
    #   all locations
    #   both sexes
    #   all causes
    #
    # for each:
    #
    #   REI + year
    # --------------------------------------------------------

    yearly = (
        df.groupby(
            [
                "rei_name",
                "year"
            ],
            as_index=False
        )["val"]
        .sum()
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    yearly = yearly.sort_values(
        [
            "rei_name",
            "year"
        ]
    )

    # --------------------------------------------------------
    # Previous year value
    # --------------------------------------------------------

    yearly["prior_year_val"] = (
        yearly
        .groupby("rei_name")["val"]
        .shift(1)
    )

    # --------------------------------------------------------
    # YoY %
    # --------------------------------------------------------

    yearly["yoy_pct_change"] = (
        (
            yearly["val"]
            - yearly["prior_year_val"]
        )
        / yearly["prior_year_val"]
        * 100
    )

    # First year has no previous year
    yearly.loc[
        yearly["prior_year_val"].isna(),
        "yoy_pct_change"
    ] = None

    return yearly


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
            r"Example:"
        )

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

    # ========================================================
    # FIND DALYs FILES
    # ========================================================

    csv_files = list(
        root.rglob("*.csv")
    )

    daly_files = []

    for csv_file in csv_files:

        measure = get_measure(
            csv_file,
            root
        )

        if measure == MEASURE:

            daly_files.append(
                csv_file
            )

    print("=" * 80)
    print("SECTION 10 — ANNUAL TREND ANALYSIS")
    print("=" * 80)

    print(
        f"Root folder : "
        f"{root.resolve()}"
    )

    print(
        f"CSV files   : "
        f"{len(csv_files)}"
    )

    print(
        f"DALYs files : "
        f"{len(daly_files)}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = load_data(
        daly_files
    )

    if df.empty:

        print()
        print(
            "No DALYs data found."
        )

        sys.exit(1)

    print()
    print(
        f"Rows loaded: "
        f"{len(df):,}"
    )

    # ========================================================
    # AVAILABLE VALUES
    # ========================================================

    print()
    print("=" * 80)
    print("AVAILABLE AGE BANDS")
    print("=" * 80)

    ages = sorted(
        df["age_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    for age in ages:

        print(
            f"  {age}"
        )

    # ========================================================
    # CHECK AGE
    # ========================================================

    if AGE not in ages:

        print()
        print(
            f"ERROR: AGE '{AGE}' "
            f"does not exist."
        )

        sys.exit(1)

    # ========================================================
    # FILTER
    # ========================================================

    filtered = df[
        df["age_name"]
        .astype(str)
        .str.strip()
        == AGE
    ]

    filtered = filtered[
        filtered["year"].between(
            START_YEAR,
            END_YEAR
        )
    ]

    print()
    print("=" * 80)
    print("FILTER / AGGREGATION")
    print("=" * 80)

    print(
        f"Measure        : {MEASURE}"
    )

    print(
        f"Metric         : {METRIC}"
    )

    print(
        f"Age            : {AGE}"
    )

    print(
        f"Years          : "
        f"{START_YEAR}-{END_YEAR}"
    )

    print(
        "Locations      : ALL 31 states/UTs"
    )

    print(
        "Sex            : Male + Female"
    )

    print(
        "Causes         : ALL CVD causes"
    )

    print(
        f"Rows selected  : "
        f"{len(filtered):,}"
    )

    if filtered.empty:

        print()
        print(
            "NO DATA AFTER FILTERING."
        )

        sys.exit(1)

    # ========================================================
    # TREND
    # ========================================================

    trend = calculate_trend(
        filtered
    )

    # ========================================================
    # ROUND VALUES
    # ========================================================

    trend["val"] = trend["val"].round(2)

    trend["prior_year_val"] = (
        trend["prior_year_val"]
        .round(2)
    )

    trend["yoy_pct_change"] = (
        trend["yoy_pct_change"]
        .round(2)
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR
        / "section_10_annual_trend_dalys.csv"
    )

    trend.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print("SECTION 10 RESULT")
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
        f"Output file:\n"
        f"{output_file.resolve()}"
    )


if __name__ == "__main__":
    main()