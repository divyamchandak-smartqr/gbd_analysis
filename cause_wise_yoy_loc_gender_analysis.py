import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# CAUSE-WISE YoY + LOCATION + GENDER ANALYSIS
# ============================================================
#
# Purpose:
#   Analyse ALL causes found across DALYs / Deaths / YLDs / YLLs.
#   A cause does NOT need to be common across all four measures.
#
#   For every cause, year and state, create:
#       - Male rows
#       - Female rows
#
#   No combined/all-gender rows are produced.
#
#   For every measure calculate:
#       - burden value
#       - YoY % change
#
#   IMPORTANT:
#   If a cause is not available for a particular measure, that
#   measure and its YoY columns are left BLANK. Missing data is
#   never converted to zero.
#
# Output:
#   ONE FINAL CSV ONLY
#
#   cause_wise_yoy_loc_gender_analysis_output/
#       cause_wise_yoy_loc_gender_analysis.csv
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2013
END_YEAR = 2023
METRIC = "Number"

MEASURES = ("DALYs", "Deaths", "YLDs", "YLLs")
SEXES = ("Male", "Female")

# Parent/heading causes that must NOT be analysed as individual diseases.
# Use normalized names so matching is case/spacing/punctuation insensitive.
EXCLUDED_PARENT_CAUSE_NAMES = {
    "cardiovasculardiseases",
}

OUTPUT_DIR = Path("cause_wise_yoy_loc_gender_analysis_output")
OUTPUT_FILE = OUTPUT_DIR / "cause_wise_yoy_loc_gender_analysis.csv"


# ============================================================
# MEASURE NORMALIZATION
# ============================================================

def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


MEASURE_ALIASES = {
    "dalys": "DALYs",
    "daly": "DALYs",
    "disabilityadjustedlifeyears": "DALYs",
    "deaths": "Deaths",
    "death": "Deaths",
    "mortality": "Deaths",
    "ylds": "YLDs",
    "yld": "YLDs",
    "yearslivedwithdisability": "YLDs",
    "ylls": "YLLs",
    "yll": "YLLs",
    "yearsoflifelost": "YLLs",
}


def normalize_measure_name(value: object) -> str | None:
    return MEASURE_ALIASES.get(normalize_token(value))


def exclude_parent_causes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove parent/header causes that are not individual disease factors."""
    out = df.copy()

    normalized_cause = out["cause_name"].map(normalize_token)
    excluded_mask = normalized_cause.isin(EXCLUDED_PARENT_CAUSE_NAMES)

    if excluded_mask.any():
        excluded_rows = out.loc[
            excluded_mask,
            ["cause_id", "cause_name"],
        ].drop_duplicates()

        print("EXCLUDING PARENT/HEADING CAUSES")
        print("-" * 100)
        print(excluded_rows.to_string(index=False))
        print()

    return out.loc[~excluded_mask].copy()


def detect_measure_from_path(file: Path, root: Path) -> str | None:
    """Detect measure from folder names or filename tokens."""
    relative = file.relative_to(root)

    for part in relative.parts:
        measure = normalize_measure_name(part)
        if measure:
            return measure

    for token in re.split(r"[_\-\s]+", file.stem):
        measure = normalize_measure_name(token)
        if measure:
            return measure

    return None


# ============================================================
# LOAD ALL VALID DATA
# ============================================================

def load_measure_data(root: Path) -> pd.DataFrame:
    print("=" * 100)
    print("CAUSE-WISE YoY + LOCATION + GENDER ANALYSIS")
    print("=" * 100)
    print()
    print(f"Root folder : {root.resolve()}")
    print(f"Years       : {START_YEAR}-{END_YEAR}")
    print(f"Metric      : {METRIC}")
    print("Sex         : Male + Female")
    print("Measures    : DALYs, Deaths, YLDs, YLLs")
    print()

    files = list(root.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root.resolve()}")

    required_base = {
        "location_name",
        "sex_name",
        "cause_id",
        "cause_name",
        "year",
        "val",
        "metric_name",
    }

    all_frames: list[pd.DataFrame] = []
    scanned = 0
    used = 0
    skipped = 0
    unreadable = 0

    for file in files:
        scanned += 1

        try:
            header = pd.read_csv(file, nrows=0)
        except Exception as exc:
            unreadable += 1
            print(f"[SKIP] Could not read header: {file}")
            print(f"       {exc}")
            continue

        columns = set(header.columns)
        missing = required_base - columns

        if missing:
            skipped += 1
            continue

        path_measure = detect_measure_from_path(file, root)
        has_measure_column = "measure_name" in columns

        # If neither path nor measure_name identifies the measure,
        # this CSV cannot be assigned safely to one of the four.
        if path_measure is None and not has_measure_column:
            skipped += 1
            continue

        usecols = list(required_base)
        if has_measure_column:
            usecols.append("measure_name")

        try:
            df = pd.read_csv(file, usecols=usecols)
        except Exception as exc:
            unreadable += 1
            print(f"[SKIP] Could not read: {file}")
            print(f"       {exc}")
            continue

        # ----------------------------------------------------
        # Measure
        # ----------------------------------------------------
        if path_measure is not None:
            df["measure"] = path_measure
        else:
            df["measure"] = df["measure_name"].map(normalize_measure_name)
            df = df[df["measure"].isin(MEASURES)]

        # ----------------------------------------------------
        # Metric = Number
        # ----------------------------------------------------
        df["metric_name"] = df["metric_name"].astype(str).str.strip()
        df = df[df["metric_name"].str.lower() == METRIC.lower()]

        # ----------------------------------------------------
        # Years
        # ----------------------------------------------------
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].between(START_YEAR, END_YEAR)]

        # ----------------------------------------------------
        # Sex: keep Male/Female source rows only.
        # No combined/all-gender rows are generated.
        # ----------------------------------------------------
        df["sex_name"] = df["sex_name"].astype(str).str.strip()
        df = df[df["sex_name"].isin(SEXES)]

        # ----------------------------------------------------
        # Cause
        # ----------------------------------------------------
        df["cause_id"] = pd.to_numeric(df["cause_id"], errors="coerce")
        df["cause_name"] = df["cause_name"].astype(str).str.strip()

        # ----------------------------------------------------
        # Value
        # ----------------------------------------------------
        df["val"] = pd.to_numeric(df["val"], errors="coerce")

        # ----------------------------------------------------
        # Location
        # ----------------------------------------------------
        df["location_name"] = df["location_name"].astype(str).str.strip()

        df = df.dropna(
            subset=[
                "cause_id",
                "cause_name",
                "year",
                "val",
                "location_name",
                "measure",
            ]
        )

        df = df[
            (df["cause_name"] != "")
            & (df["cause_name"].str.lower() != "nan")
            & (df["location_name"] != "")
            & (df["location_name"].str.lower() != "nan")
        ]

        if df.empty:
            continue

        df["cause_id"] = df["cause_id"].astype("int64")
        df["year"] = df["year"].astype("int64")

        all_frames.append(
            df[
                [
                    "measure",
                    "cause_id",
                    "cause_name",
                    "location_name",
                    "sex_name",
                    "year",
                    "val",
                ]
            ].copy()
        )
        used += 1

    if not all_frames:
        raise ValueError("No valid rows remained after filtering input files.")

    combined = pd.concat(all_frames, ignore_index=True)

    print(f"CSV files scanned : {scanned:,}")
    print(f"CSV files used    : {used:,}")
    print(f"CSV files skipped : {skipped:,}")
    print(f"Unreadable files  : {unreadable:,}")
    print(f"Rows loaded       : {len(combined):,}")
    print()

    return combined


# ============================================================
# CANONICAL CAUSE MASTER — UNION ACROSS ALL FOUR MEASURES
# ============================================================

def build_cause_master(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("BUILDING ALL-CAUSE MASTER")
    print("=" * 100)

    names: dict[int, Counter[str]] = defaultdict(Counter)
    presence: dict[int, set[str]] = defaultdict(set)

    unique_pairs = df[
        ["measure", "cause_id", "cause_name"]
    ].drop_duplicates()

    # Use explicit column lists instead of itertuples() here.
    # This avoids pandas/Pylance NamedTuple inference conflicts while
    # preserving the exact same runtime logic.
    measures_list = unique_pairs["measure"].astype(str).tolist()
    cause_ids_list = unique_pairs["cause_id"].tolist()
    cause_names_list = unique_pairs["cause_name"].astype(str).tolist()

    for measure_value, cause_id_value, cause_name_value in zip(
        measures_list,
        cause_ids_list,
        cause_names_list,
    ):
        measure = str(measure_value)
        cause_id = int(float(str(cause_id_value)))
        cause_name = str(cause_name_value).strip()

        names[cause_id][cause_name] += 1
        presence[cause_id].add(measure)

    rows: list[dict[str, object]] = []

    for cause_id in sorted(names):
        canonical_name = sorted(
            names[cause_id].items(),
            key=lambda item: (-item[1], item[0].lower()),
        )[0][0]

        row: dict[str, object] = {
            "Cause ID": cause_id,
            "Cause Name": canonical_name,
        }

        for measure in MEASURES:
            row[measure] = measure in presence[cause_id]

        rows.append(row)

    master = pd.DataFrame(rows)

    print(f"All unique causes : {len(master):,}")
    for measure in MEASURES:
        count = int(master[measure].sum())
        print(f"{measure:<8}: {count:,}")
    print()

    return master


# ============================================================
# NORMALIZE CAUSE NAMES TO CANONICAL MASTER NAME
# ============================================================

def apply_canonical_cause_names(
    df: pd.DataFrame,
    cause_master: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    name_map = dict(
        zip(
            cause_master["Cause ID"].astype(int),
            cause_master["Cause Name"].astype(str),
        )
    )

    out["cause_name"] = out["cause_id"].map(name_map)

    return out


# ============================================================
# AGGREGATE STATE + GENDER — MALE / FEMALE ONLY
# ============================================================

def build_long_analysis(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("CALCULATING STATE + GENDER BURDEN — MALE / FEMALE ONLY")
    print("=" * 100)

    # Sum over all remaining source dimensions while keeping:
    # Measure + Cause + State + Gender + Year.
    state_gender = (
        df.groupby(
            [
                "measure",
                "cause_id",
                "cause_name",
                "location_name",
                "sex_name",
                "year",
            ],
            as_index=False,
        )
        .agg(value=("val", "sum"))
    )

    state_gender["gender"] = state_gender["sex_name"]
    state_gender = state_gender.drop(columns=["sex_name"])

    state_gender = state_gender[
        [
            "measure",
            "cause_id",
            "cause_name",
            "location_name",
            "gender",
            "year",
            "value",
        ]
    ]

    print(f"Long analysis rows : {len(state_gender):,}")
    print("Gender rows        : Male + Female only")
    print()

    return state_gender


# ============================================================
# YoY % CHANGE
# ============================================================

def add_yoy(long_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("CALCULATING YEAR-ON-YEAR % CHANGE")
    print("=" * 100)

    out = long_df.copy()

    group_keys = [
        "measure",
        "cause_id",
        "cause_name",
        "location_name",
        "gender",
    ]

    out = out.sort_values(group_keys + ["year"]).reset_index(drop=True)

    out["previous_year"] = out.groupby(group_keys)["year"].shift(1)
    out["previous_value"] = out.groupby(group_keys)["value"].shift(1)

    out["yoy_pct"] = pd.NA

    valid = (
        (out["previous_year"] == out["year"] - 1)
        & out["previous_value"].notna()
        & (out["previous_value"] != 0)
    )

    out.loc[valid, "yoy_pct"] = (
        (
            out.loc[valid, "value"]
            - out.loc[valid, "previous_value"]
        )
        / out.loc[valid, "previous_value"]
        * 100
    )

    print("YoY % = (Current Year - Previous Year) / Previous Year × 100")
    print("2013 YoY remains blank because 2012 is outside the analysis period.")
    print()

    return out


# ============================================================
# FINAL COMPLETE GRID — KEEPS MISSING MEASURES BLANK
# ============================================================

def build_complete_grid(
    long_df: pd.DataFrame,
    cause_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create every expected Cause x State x Gender x Year x Measure
    combination, then left join real values.

    This guarantees that a cause absent from one measure remains
    present in the final CSV and that unavailable measure cells stay
    blank instead of being converted to zero.
    """
    print("=" * 100)
    print("BUILDING COMPLETE ALL-CAUSE MALE/FEMALE GRID")
    print("=" * 100)

    cause_location_rows = (
        long_df[["cause_id", "cause_name", "location_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    gender_rows = pd.DataFrame({"gender": list(SEXES)})
    years = pd.DataFrame({"year": list(range(START_YEAR, END_YEAR + 1))})
    measures = pd.DataFrame({"measure": list(MEASURES)})

    grid = (
        cause_location_rows.merge(gender_rows, how="cross")
        .merge(years, how="cross")
        .merge(measures, how="cross")
    )

    value_cols = [
        "measure",
        "cause_id",
        "cause_name",
        "location_name",
        "gender",
        "year",
        "value",
        "yoy_pct",
    ]

    merged = grid.merge(
        long_df[value_cols],
        on=[
            "measure",
            "cause_id",
            "cause_name",
            "location_name",
            "gender",
            "year",
        ],
        how="left",
        validate="one_to_one",
    )

    print(f"Complete grid rows : {len(merged):,}")
    print()

    return merged


# ============================================================
# FORMATTERS
# ============================================================

def to_optional_float(value: object) -> float | None:
    """Convert a pandas/object scalar to float without Pylance overload issues."""
    if value is None or value is pd.NA:
        return None

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "nat", "none", "<na>"}:
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def format_burden(value: object) -> str:
    number = to_optional_float(value)

    if number is None:
        return ""

    absolute = abs(number)

    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"{number / 1_000:.2f}K"

    return f"{number:.2f}"


def format_yoy(value: object) -> str:
    number = to_optional_float(value)

    if number is None:
        return ""

    if number > 0:
        return f"+{number:.2f}%"

    return f"{number:.2f}%"


# ============================================================
# CREATE ONE FINAL WIDE CSV
# ============================================================

def create_final_table(complete: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("CREATING ONE FINAL WIDE CSV")
    print("=" * 100)

    base_keys = [
        "cause_id",
        "cause_name",
        "location_name",
        "gender",
    ]

    rows: list[dict[str, object]] = []

    for keys, group in complete.groupby(base_keys, sort=False):
        cause_id, cause_name, location_name, gender = keys

        row: dict[str, object] = {
            "Cause ID": int(float(str(cause_id))),
            "Cause Name": str(cause_name),
            "State": str(location_name),
            "Gender": str(gender),
        }

        # Explicitly build a typed lookup from columns instead of using
        # pandas NamedTuple attributes. This keeps Pylance from treating
        # year/value/yoy_pct as overly broad Scalar/object types.
        lookup: dict[tuple[str, int], tuple[object, object]] = {}

        measure_values = group["measure"].astype(str).tolist()
        year_values = group["year"].tolist()
        burden_values = group["value"].tolist()
        yoy_values = group["yoy_pct"].tolist()

        for measure_value, year_value, burden_value, yoy_value in zip(
            measure_values,
            year_values,
            burden_values,
            yoy_values,
        ):
            year_key = int(float(str(year_value)))
            lookup[(str(measure_value), year_key)] = (
                burden_value,
                yoy_value,
            )

        for year in range(START_YEAR, END_YEAR + 1):
            for measure in MEASURES:
                current = lookup.get((measure, year))

                if current is None:
                    row[f"{year} {measure}"] = ""
                    row[f"{year} {measure} YoY %"] = ""
                    continue

                burden_value, yoy_value = current
                row[f"{year} {measure}"] = format_burden(burden_value)
                row[f"{year} {measure} YoY %"] = format_yoy(yoy_value)

        rows.append(row)

    final = pd.DataFrame(rows)

    columns = [
        "Cause ID",
        "Cause Name",
        "State",
        "Gender",
    ]

    for year in range(START_YEAR, END_YEAR + 1):
        for measure in MEASURES:
            columns.extend(
                [
                    f"{year} {measure}",
                    f"{year} {measure} YoY %",
                ]
            )

    final = final[columns]

    gender_order = {
        "Male": 0,
        "Female": 1,
    }

    final["_gender_order"] = final["Gender"].map(gender_order)

    final = final.sort_values(
        [
            "Cause ID",
            "State",
            "_gender_order",
        ],
        ascending=[True, True, True],
        na_position="last",
    )

    final = final.drop(columns=["_gender_order"])
    final = final.reset_index(drop=True)

    print(f"Final rows    : {len(final):,}")
    print(f"Final columns : {len(final.columns):,}")
    print()

    return final


# ============================================================
# VALIDATION
# ============================================================

def validate_final(final: pd.DataFrame, cause_master: pd.DataFrame) -> None:
    expected_causes = set(cause_master["Cause ID"].astype(int))
    actual_causes = set(final["Cause ID"].astype(int))

    if expected_causes != actual_causes:
        missing = sorted(expected_causes - actual_causes)
        extra = sorted(actual_causes - expected_causes)
        raise ValueError(
            f"Cause validation failed. Missing={missing}; Extra={extra}"
        )

    duplicate_keys = [
        "Cause ID",
        "State",
        "Gender",
    ]

    if final.duplicated(subset=duplicate_keys).any():
        raise ValueError("Duplicate final analysis rows detected.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python cause_wise_yoy_loc_gender_analysis.py "
            "<normalise_gbd_dataset>"
        )
        print()
        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():
        print(f"Folder not found: {root.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all four measures.
    source = load_measure_data(root)

    # 2. Remove parent/header causes.
    #    Example: "Cardiovascular diseases" is a broad heading and
    #    must not be analysed as if it were an individual disease.
    source = exclude_parent_causes(source)

    # 3. Build UNION of all remaining individual cause IDs/names.
    cause_master = build_cause_master(source)

    # 4. Normalize names by Cause ID.
    source = apply_canonical_cause_names(source, cause_master)

    # 5. Build State + Gender values for Male and Female only.
    long_analysis = build_long_analysis(source)

    # 6. Calculate YoY separately for each measure/cause/state/gender.
    long_analysis = add_yoy(long_analysis)

    # 7. Complete grid keeps unavailable cause/measure combinations blank.
    complete = build_complete_grid(long_analysis, cause_master)

    # 8. Create years-as-headings final table.
    final = create_final_table(complete)

    # 9. Validate all discovered individual causes are retained.
    validate_final(final, cause_master)

    # 10. SAVE ONLY ONE FINAL CSV.
    final.to_csv(OUTPUT_FILE, index=False)

    print("=" * 100)
    print("CAUSE-WISE ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print(f"Unique causes analysed : {final['Cause ID'].nunique():,}")
    print(f"States                 : {final['State'].nunique():,}")
    print(f"Years                  : {START_YEAR}-{END_YEAR}")
    print("Gender                 : Male + Female only")
    print("Missing measure data   : BLANK")
    print("Final files            : 1")
    print()
    print("Final CSV:")
    print(OUTPUT_FILE.resolve())
    print()

    # Print the cause master summary to terminal without creating
    # another file.
    print("ALL INDIVIDUAL CAUSES USED")
    print("-" * 100)
    master_print = cause_master.copy()
    for measure in MEASURES:
        master_print[measure] = master_print[measure].map(
            {True: "Yes", False: "No"}
        )
    print(master_print.to_string(index=False))


if __name__ == "__main__":
    main()