import sys
from pathlib import Path

import pandas as pd


# ============================================================
# STEP 15 — RISK-FACTOR TREND RANKING
# ============================================================
#
# Purpose:
#   Combine annual risk-factor burden trends with each risk
#   factor's yearly contribution (Current Weight %) to assign
#   a yearly Priority.
#
# Years:
#   2013–2023
#
# REIs:
#   20 common REIs
#
# Measure:
#   DALYs
#
# Metric:
#   Number
#
# Final CSV layout:
#   Risk Factor | REI ID |
#   2013 Current Weight | 2013 YoY % Change | 2013 Trend | 2013 Priority |
#   2014 Current Weight | 2014 YoY % Change | 2014 Trend | 2014 Priority |
#   ...
#   2023 Current Weight | 2023 YoY % Change | 2023 Trend | 2023 Priority
#
# Output:
#   ONE final CSV only
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

METRIC = "Number"
START_YEAR = 2013
END_YEAR = 2023

OUTPUT_DIR = Path("step_15_output")
OUTPUT_FILE = OUTPUT_DIR / "step_15_risk_factor_trend_ranking.csv"

# Step 10 trend classification threshold.
# YoY > +1.0%  -> Increasing
# YoY < -1.0%  -> Decreasing
# otherwise    -> Stable
#
# Keep this equal to the threshold used in Step 10.
STABLE_THRESHOLD_PCT = 1.0


# ============================================================
# PRIORITY WEIGHT BANDS
# ============================================================
# These rules reproduce the Step 15 interpretation discussed:
#
# High SBP  24% + Increasing -> Very High
# Smoking   14% + Stable     -> High
# High BMI  12% + Increasing -> High
# Alcohol    5% + Decreasing -> Moderate
#
# The bands are configurable here if you later want to adjust
# the priority framework without changing the rest of the code.
# ============================================================

VERY_HIGH_WEIGHT_THRESHOLD = 20.0
HIGH_WEIGHT_THRESHOLD = 10.0
MODERATE_WEIGHT_THRESHOLD = 5.0


# ============================================================
# 20 COMMON REIs
# ============================================================

COMMON_REI_IDS = {
    91,
    99,
    100,
    102,
    105,
    107,
    108,
    116,
    117,
    118,
    119,
    121,
    122,
    123,
    124,
    125,
    332,
    341,
    367,
    380,
}


# ============================================================
# FIND DALYs FILES
# ============================================================

def find_dalys_files(root: Path) -> list[Path]:
    files = []

    for file in root.rglob("*.csv"):
        relative_parts = [
            part.lower()
            for part in file.relative_to(root).parts
        ]

        # Same folder-based DALYs discovery style used in the
        # earlier state-wise analysis.
        if "dalys" in relative_parts:
            files.append(file)

    return files


# ============================================================
# LOAD + FILTER DALYs DATA
# ============================================================

def load_dalys(root: Path) -> pd.DataFrame:
    files = find_dalys_files(root)

    print("=" * 90)
    print("STEP 15 — RISK-FACTOR TREND RANKING")
    print("=" * 90)
    print()
    print(f"Root folder : {root.resolve()}")
    print(f"DALYs files : {len(files)}")
    print(f"Metric      : {METRIC}")
    print(f"Years       : {START_YEAR}-{END_YEAR}")
    print("REIs        : 20 COMMON REIs")
    print("Output      : ONE FINAL CSV")
    print()

    if not files:
        print("ERROR: No DALYs CSV files found.")
        sys.exit(1)

    all_data = []

    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:
            print(f"[SKIP] Could not read: {file}")
            print(exc)
            continue

        required_columns = [
            "rei_id",
            "rei_name",
            "year",
            "val",
            "metric_name",
        ]

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            print(f"[SKIP] Missing columns in: {file}")
            print("Missing:", missing)
            continue

        # ----------------------------------------------------
        # METRIC
        # ----------------------------------------------------
        df["metric_name"] = (
            df["metric_name"]
            .astype(str)
            .str.strip()
        )

        df = df[
            df["metric_name"].str.lower() == METRIC.lower()
        ]

        # ----------------------------------------------------
        # YEAR
        # ----------------------------------------------------
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        )

        df = df[
            df["year"].between(START_YEAR, END_YEAR)
        ]

        # ----------------------------------------------------
        # REI
        # ----------------------------------------------------
        df["rei_id"] = pd.to_numeric(
            df["rei_id"],
            errors="coerce",
        )

        df = df[
            df["rei_id"].isin(COMMON_REI_IDS)
        ]

        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------
        df["val"] = pd.to_numeric(
            df["val"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["rei_id", "rei_name", "year", "val"]
        )

        if df.empty:
            continue

        df["rei_name"] = (
            df["rei_name"]
            .astype(str)
            .str.strip()
        )

        all_data.append(
            df[["rei_id", "rei_name", "year", "val"]].copy()
        )

    if not all_data:
        print("ERROR: No valid DALYs rows remained after filtering.")
        sys.exit(1)

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    print(f"Rows loaded : {len(combined):,}")
    print()

    return combined


# ============================================================
# CALCULATE YEARLY RISK-FACTOR BURDEN
# ============================================================

def calculate_yearly_burden(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CALCULATING YEARLY RISK-FACTOR BURDEN")
    print("=" * 90)

    yearly = (
        df.groupby(
            ["rei_id", "rei_name", "year"],
            as_index=False,
        )["val"]
        .sum()
        .rename(columns={"val": "burden"})
    )

    yearly = yearly.sort_values(
        ["rei_id", "year"],
        ascending=[True, True],
    ).reset_index(drop=True)

    print(f"Yearly rows : {len(yearly):,}")
    print()

    return yearly


# ============================================================
# CURRENT WEIGHT %
# ============================================================

def calculate_current_weight(yearly: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CALCULATING CURRENT WEIGHT %")
    print("=" * 90)

    yearly = yearly.copy()

    yearly["year_total_burden"] = (
        yearly.groupby("year")["burden"]
        .transform("sum")
    )

    yearly["current_weight_pct"] = 0.0

    valid_total = yearly["year_total_burden"] != 0

    yearly.loc[
        valid_total,
        "current_weight_pct",
    ] = (
        yearly.loc[valid_total, "burden"]
        / yearly.loc[valid_total, "year_total_burden"]
        * 100
    )

    print("Current Weight = Risk burden / total selected-risk burden for that year")
    print()

    return yearly


# ============================================================
# STEP 10 YoY % CHANGE
# ============================================================

def calculate_yoy(yearly: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CALCULATING YoY % CHANGE — STEP 10 LOGIC")
    print("=" * 90)

    yearly = yearly.copy()
    yearly = yearly.sort_values(
        ["rei_id", "rei_name", "year"]
    ).reset_index(drop=True)

    yearly["previous_year"] = (
        yearly.groupby(["rei_id", "rei_name"])["year"]
        .shift(1)
    )

    yearly["previous_year_burden"] = (
        yearly.groupby(["rei_id", "rei_name"])["burden"]
        .shift(1)
    )

    yearly["yoy_pct_change"] = pd.NA

    immediate_previous_year = (
        yearly["previous_year"] == yearly["year"] - 1
    )

    valid_denominator = (
        yearly["previous_year_burden"].notna()
        & (yearly["previous_year_burden"] != 0)
    )

    valid_yoy = immediate_previous_year & valid_denominator

    yearly.loc[
        valid_yoy,
        "yoy_pct_change",
    ] = (
        (
            yearly.loc[valid_yoy, "burden"]
            - yearly.loc[valid_yoy, "previous_year_burden"]
        )
        / yearly.loc[valid_yoy, "previous_year_burden"]
        * 100
    )

    print("YoY % = (Current Year - Previous Year) / Previous Year × 100")
    print()

    return yearly


# ============================================================
# TREND CLASSIFICATION
# ============================================================

def classify_trend(year: int, yoy_value) -> str:
    if int(year) == START_YEAR or pd.isna(yoy_value):
        return "Baseline"

    yoy = float(yoy_value)

    if yoy > STABLE_THRESHOLD_PCT:
        return "Increasing"

    if yoy < -STABLE_THRESHOLD_PCT:
        return "Decreasing"

    return "Stable"


def add_trend(yearly: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CLASSIFYING YEARLY TREND")
    print("=" * 90)

    yearly = yearly.copy()

    yearly["trend"] = yearly.apply(
        lambda row: classify_trend(
            row["year"],
            row["yoy_pct_change"],
        ),
        axis=1,
    )

    print(f"Stable range : ±{STABLE_THRESHOLD_PCT:.2f}%")
    print()

    return yearly


# ============================================================
# PRIORITY CLASSIFICATION
# ============================================================

def classify_priority(weight_pct: float, trend: str) -> str:
    """
    Priority combines Current Weight + Step 10 Trend.

    Rules are intentionally aligned with the discussed example:

      24% + Increasing -> Very High
      14% + Stable     -> High
      12% + Increasing -> High
       5% + Decreasing -> Moderate
    """

    weight = float(weight_pct)

    # Baseline year has no YoY trend yet, so priority is based
    # only on current contribution.
    if trend == "Baseline":
        if weight >= VERY_HIGH_WEIGHT_THRESHOLD:
            return "High"
        if weight >= HIGH_WEIGHT_THRESHOLD:
            return "High"
        if weight >= MODERATE_WEIGHT_THRESHOLD:
            return "Moderate"
        return "Low"

    # Very high priority requires both very high contribution
    # and an increasing burden.
    if (
        weight >= VERY_HIGH_WEIGHT_THRESHOLD
        and trend == "Increasing"
    ):
        return "Very High"

    # A risk carrying at least 10% of selected burden remains
    # high priority regardless of short-term direction.
    if weight >= HIGH_WEIGHT_THRESHOLD:
        return "High"

    # Moderate current contribution becomes high priority when
    # it is increasing.
    if weight >= MODERATE_WEIGHT_THRESHOLD:
        if trend == "Increasing":
            return "High"
        return "Moderate"

    # Lower-weight risks still receive Moderate priority when
    # increasing, otherwise Low.
    if trend == "Increasing":
        return "Moderate"

    return "Low"


def add_priority(yearly: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CALCULATING YEARLY PRIORITY")
    print("=" * 90)

    yearly = yearly.copy()

    yearly["priority"] = yearly.apply(
        lambda row: classify_priority(
            row["current_weight_pct"],
            row["trend"],
        ),
        axis=1,
    )

    print("Priority = Current Weight + Trend")
    print()

    return yearly


# ============================================================
# FORMATTERS
# ============================================================

def format_weight(value) -> str:
    if pd.isna(value):
        return ""

    return f"{float(value):.2f}%"


def format_yoy(value) -> str:
    if pd.isna(value):
        return ""

    value = float(value)

    if value > 0:
        return f"+{value:.2f}%"

    return f"{value:.2f}%"


# ============================================================
# CREATE FINAL WIDE TABLE
# ============================================================

def create_final_table(yearly: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("CREATING FINAL WIDE TABLE")
    print("=" * 90)

    rows = []

    for (rei_id, rei_name), group in yearly.groupby(
        ["rei_id", "rei_name"]
    ):
        group = group.sort_values("year")

        row = {
            "Risk Factor": rei_name,
        }

        lookup = {
            int(item["year"]): item
            for _, item in group.iterrows()
        }

        for year in range(START_YEAR, END_YEAR + 1):
            current = lookup.get(year)

            if current is None:
                row[f"{year} Current Weight"] = ""
                row[f"{year} YoY % Change"] = ""
                row[f"{year} Trend"] = ""
                row[f"{year} Priority"] = ""
                continue

            row[f"{year} Current Weight"] = format_weight(
                current["current_weight_pct"]
            )

            row[f"{year} YoY % Change"] = format_yoy(
                current["yoy_pct_change"]
            )

            row[f"{year} Trend"] = current["trend"]
            row[f"{year} Priority"] = current["priority"]

        # Internal numeric field used only for deterministic row
        # ordering. It is removed before saving the CSV.
        latest = lookup.get(END_YEAR)
        row["_latest_weight"] = (
            float(latest["current_weight_pct"])
            if latest is not None
            else float("nan")
        )
        row["_rei_id"] = int(rei_id)

        rows.append(row)

    final = pd.DataFrame(rows)

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------
    columns = [
        "Risk Factor",
    ]

    for year in range(START_YEAR, END_YEAR + 1):
        columns.extend(
            [
                f"{year} Current Weight",
                f"{year} YoY % Change",
                f"{year} Trend",
                f"{year} Priority",
            ]
        )

    # --------------------------------------------------------
    # ROW ORDER
    # --------------------------------------------------------
    # A single wide CSV cannot have a different row order for
    # every year, so rows are ordered by latest-year (2023)
    # Current Weight descending, then REI ID.
    # --------------------------------------------------------
    final = final.sort_values(
        ["_latest_weight", "_rei_id"],
        ascending=[False, True],
        na_position="last",
    )

    final = final.drop(columns=["_latest_weight", "_rei_id"])
    final = final[columns].reset_index(drop=True)

    print(f"Final risk factors : {len(final):,}")
    print()

    return final


# ============================================================
# VALIDATION
# ============================================================

def validate_output(final: pd.DataFrame) -> None:
    expected_year_columns = 4 * (
        END_YEAR - START_YEAR + 1
    )

    expected_columns = 1 + expected_year_columns

    if len(final.columns) != expected_columns:
        raise ValueError(
            "Unexpected final column count: "
            f"{len(final.columns)} != {expected_columns}"
        )

    if final["Risk Factor"].duplicated().any():
        duplicates = final.loc[
            final["Risk Factor"].duplicated(keep=False),
            ["Risk Factor"],
        ]

        raise ValueError(
            "Duplicate risk-factor rows found in final output:\n"
            + duplicates.to_string(index=False)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python step_15_risk_factor_trend_ranking.py "
            "<normalised_gbd_dataset>"
        )
        print()
        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():
        print(f"Folder not found: {root.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # STEP 1 — Load filtered DALYs data
    df = load_dalys(root)

    # STEP 2 — Risk-factor burden for every year
    yearly = calculate_yearly_burden(df)

    # STEP 3 — Current Weight % for every year
    yearly = calculate_current_weight(yearly)

    # STEP 4 — YoY % Change using Step 10 logic
    yearly = calculate_yoy(yearly)

    # STEP 5 — Increasing / Stable / Decreasing
    yearly = add_trend(yearly)

    # STEP 6 — Priority from Current Weight + Trend
    yearly = add_priority(yearly)

    # STEP 7 — Years as headings in one wide final table
    final = create_final_table(yearly)

    # STEP 8 — Validate final structure
    validate_output(final)

    # STEP 9 — Save ONLY ONE CSV
    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 90)
    print("STEP 15 COMPLETE")
    print("=" * 90)
    print()
    print(f"Risk factors  : {final['Risk Factor'].nunique()}")
    print(f"Years         : {START_YEAR}-{END_YEAR}")
    print("Final files   : 1")
    print()
    print("Final CSV:")
    print(OUTPUT_FILE.resolve())
    print()
    print("Final columns per year:")
    print("  Current Weight")
    print("  YoY % Change")
    print("  Trend")
    print("  Priority")
    print()
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()