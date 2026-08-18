import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ============================================================
# STEP 16 — CAUSE MASTER + COMMON CAUSES ACROSS 4 MEASURES
# ============================================================
#
# Purpose:
#   1. Find every unique cause_id + cause_name available in:
#        - DALYs
#        - Deaths
#        - YLDs
#        - YLLs
#   2. Show whether each cause is present in each measure.
#   3. Identify causes common to ALL FOUR measures.
#
# Output:
#   ONE CSV only:
#       step_16_output/step_16_cause_master.csv
#
# Final columns:
#   Cause ID
#   Cause Name
#   DALYs
#   Deaths
#   YLDs
#   YLLs
#   Common Across All 4
#
# ============================================================


OUTPUT_DIR = Path("step_16_output")
OUTPUT_FILE = OUTPUT_DIR / "step_16_cause_master.csv"

MEASURES = ("DALYs", "Deaths", "YLDs", "YLLs")


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_token(value: object) -> str:
    """Normalize text for reliable measure matching."""
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
    token = normalize_token(value)
    return MEASURE_ALIASES.get(token)


def detect_measure_from_path(file: Path, root: Path) -> str | None:
    """Detect DALYs / Deaths / YLDs / YLLs from folder or filename."""
    relative = file.relative_to(root)

    # Check path components first because normalized datasets are often
    # organised into measure folders.
    for part in relative.parts:
        measure = normalize_measure_name(part)
        if measure:
            return measure

    # Then inspect filename tokens such as DALYs_Number_....csv
    stem_tokens = re.split(r"[_\-\s]+", file.stem)
    for token in stem_tokens:
        measure = normalize_measure_name(token)
        if measure:
            return measure

    return None


# ============================================================
# CAUSE COLLECTION
# ============================================================

def clean_cause_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid, normalized cause_id + cause_name rows."""
    out = df[["cause_id", "cause_name"]].copy()

    out["cause_id"] = pd.to_numeric(
        out["cause_id"],
        errors="coerce",
    )

    out["cause_name"] = (
        out["cause_name"]
        .astype(str)
        .str.strip()
    )

    out = out.dropna(subset=["cause_id"])
    out = out[
        out["cause_name"].notna()
        & (out["cause_name"] != "")
        & (out["cause_name"].str.lower() != "nan")
    ]

    out["cause_id"] = out["cause_id"].astype("int64")

    return out.drop_duplicates(
        subset=["cause_id", "cause_name"]
    ).reset_index(drop=True)


def collect_causes(root: Path):
    print("=" * 90)
    print("STEP 16 — FINDING CAUSES ACROSS DALYs / DEATHS / YLDs / YLLs")
    print("=" * 90)
    print()
    print(f"Root folder : {root.resolve()}")
    print()

    # measure -> cause_id -> Counter(cause_name)
    names_by_measure: dict[str, dict[int, Counter[str]]] = {
        measure: defaultdict(Counter)
        for measure in MEASURES
    }

    scanned_files = 0
    used_files = 0
    skipped_missing_columns = 0
    unreadable_files = 0

    csv_files = list(root.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under: {root.resolve()}"
        )

    for file in csv_files:
        scanned_files += 1

        try:
            header = pd.read_csv(file, nrows=0)
        except Exception as exc:
            unreadable_files += 1
            print(f"[SKIP] Could not read header: {file}")
            print(f"       {exc}")
            continue

        columns = set(header.columns)

        if not {"cause_id", "cause_name"}.issubset(columns):
            skipped_missing_columns += 1
            continue

        path_measure = detect_measure_from_path(file, root)

        # ----------------------------------------------------
        # Case A: measure is known from path / filename
        # ----------------------------------------------------
        if path_measure:
            try:
                df = pd.read_csv(
                    file,
                    usecols=["cause_id", "cause_name"],
                )
            except Exception as exc:
                unreadable_files += 1
                print(f"[SKIP] Could not read: {file}")
                print(f"       {exc}")
                continue

            causes = clean_cause_rows(df)

            for row in causes.itertuples(index=False):
                cause_id = int(row[0])
                cause_name = str(row[1])
                names_by_measure[path_measure][cause_id][cause_name] += 1

            used_files += 1
            continue

        # ----------------------------------------------------
        # Case B: measure is stored in measure_name column
        # ----------------------------------------------------
        if "measure_name" in columns:
            try:
                df = pd.read_csv(
                    file,
                    usecols=["cause_id", "cause_name", "measure_name"],
                )
            except Exception as exc:
                unreadable_files += 1
                print(f"[SKIP] Could not read: {file}")
                print(f"       {exc}")
                continue

            df["_measure"] = df["measure_name"].map(
                normalize_measure_name
            )

            df = df[df["_measure"].isin(MEASURES)]

            if df.empty:
                continue

            for measure, measure_df in df.groupby("_measure"):
                measure_name = str(measure)
                causes = clean_cause_rows(measure_df)

                for row in causes.itertuples(index=False):
                    cause_id = int(row[0])
                    cause_name = str(row[1])
                    names_by_measure[measure_name][cause_id][cause_name] += 1

            used_files += 1

    print(f"CSV files scanned          : {scanned_files:,}")
    print(f"CSV files used             : {used_files:,}")
    print(f"Missing cause columns      : {skipped_missing_columns:,}")
    print(f"Unreadable CSV files       : {unreadable_files:,}")
    print()

    return names_by_measure


# ============================================================
# BUILD ONE CAUSE MASTER
# ============================================================

def choose_canonical_name(
    cause_id: int,
    names_by_measure: dict[str, dict[int, Counter[str]]],
) -> str:
    """Choose the most frequently observed name for the cause ID."""
    combined_names: Counter[str] = Counter()

    for measure in MEASURES:
        combined_names.update(
            names_by_measure[measure].get(cause_id, Counter())
        )

    if not combined_names:
        return ""

    # Highest occurrence first; alphabetical name breaks ties.
    return sorted(
        combined_names.items(),
        key=lambda item: (-item[1], item[0].lower()),
    )[0][0]


def build_cause_master(
    names_by_measure: dict[str, dict[int, Counter[str]]],
) -> pd.DataFrame:
    all_cause_ids: set[int] = set()

    for measure in MEASURES:
        all_cause_ids.update(names_by_measure[measure].keys())

    rows = []

    for cause_id in sorted(all_cause_ids):
        presence = {
            measure: cause_id in names_by_measure[measure]
            for measure in MEASURES
        }

        common_all_four = all(presence.values())

        rows.append(
            {
                "Cause ID": cause_id,
                "Cause Name": choose_canonical_name(
                    cause_id,
                    names_by_measure,
                ),
                "DALYs": "Yes" if presence["DALYs"] else "No",
                "Deaths": "Yes" if presence["Deaths"] else "No",
                "YLDs": "Yes" if presence["YLDs"] else "No",
                "YLLs": "Yes" if presence["YLLs"] else "No",
                "Common Across All 4": (
                    "Yes" if common_all_four else "No"
                ),
            }
        )

    final = pd.DataFrame(rows)

    if final.empty:
        return final

    # Common causes first, then Cause ID ascending.
    final["_common_sort"] = (
        final["Common Across All 4"] == "Yes"
    ).astype(int)

    final = final.sort_values(
        ["_common_sort", "Cause ID"],
        ascending=[False, True],
    ).drop(columns=["_common_sort"])

    return final.reset_index(drop=True)


# ============================================================
# VALIDATION + SUMMARY
# ============================================================

def validate_master(final: pd.DataFrame) -> None:
    expected_columns = [
        "Cause ID",
        "Cause Name",
        "DALYs",
        "Deaths",
        "YLDs",
        "YLLs",
        "Common Across All 4",
    ]

    if list(final.columns) != expected_columns:
        raise ValueError(
            "Unexpected final columns:\n"
            f"Expected: {expected_columns}\n"
            f"Actual  : {list(final.columns)}"
        )

    if final["Cause ID"].duplicated().any():
        duplicates = final.loc[
            final["Cause ID"].duplicated(keep=False),
            ["Cause ID", "Cause Name"],
        ]
        raise ValueError(
            "Duplicate Cause IDs found:\n"
            + duplicates.to_string(index=False)
        )


def print_summary(
    final: pd.DataFrame,
    names_by_measure: dict[str, dict[int, Counter[str]]],
) -> None:
    print("=" * 90)
    print("STEP 16 CAUSE DISCOVERY COMPLETE")
    print("=" * 90)
    print()

    for measure in MEASURES:
        print(
            f"{measure:<8}: "
            f"{len(names_by_measure[measure]):,} unique causes"
        )

    common = final[
        final["Common Across All 4"] == "Yes"
    ]

    print()
    print(f"All unique Cause IDs       : {len(final):,}")
    print(f"Common across all 4        : {len(common):,}")
    print()
    print("Final CSV:")
    print(OUTPUT_FILE.resolve())
    print()

    if not common.empty:
        print("COMMON CAUSES ACROSS DALYs / DEATHS / YLDs / YLLs")
        print("-" * 90)
        print(
            common[["Cause ID", "Cause Name"]]
            .to_string(index=False)
        )
    else:
        print("No causes were common across all four measures.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python step_16_find_common_causes.py "
            "<normalised_gbd_dataset>"
        )
        print()
        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():
        print(f"Folder not found: {root.resolve()}")
        sys.exit(1)

    names_by_measure = collect_causes(root)
    final = build_cause_master(names_by_measure)

    if final.empty:
        print("ERROR: No causes were found in the four target measures.")
        sys.exit(1)

    validate_master(final)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save exactly ONE CSV for this discovery step.
    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(final, names_by_measure)


if __name__ == "__main__":
    main()