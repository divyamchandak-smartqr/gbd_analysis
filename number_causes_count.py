import os
import sys
import re
from pathlib import Path

import pandas as pd


# ============================================================
# ALL-CAUSE COUNT ANALYSIS
# ============================================================

START_YEAR = 2013
END_YEAR = 2023

MEASURES = (
    "DALYs",
    "Deaths",
    "YLLs",
    "YLDs",
)

METRIC = "Number"

OUTPUT_DIR = Path(
    "all_cause_parameter_count_output"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "all_cause_parameter_count.csv"
)


# ============================================================
# MEASURE NORMALIZATION
# ============================================================

def normalize_token(value):
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower()
    )


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


def normalize_measure_name(value):

    return MEASURE_ALIASES.get(
        normalize_token(value)
    )


# ============================================================
# DETECT MEASURE FROM FILE PATH
# ============================================================

def detect_measure_from_path(
    file_path,
    root
):

    relative = file_path.relative_to(root)

    # Check folder names first
    for part in relative.parts:

        measure = normalize_measure_name(part)

        if measure:
            return measure

    # Check filename
    for token in re.split(
        r"[_\-\s]+",
        file_path.stem
    ):

        measure = normalize_measure_name(token)

        if measure:
            return measure

    return None


# ============================================================
# LOAD DATA AND COUNT RECORDS
# ============================================================

def main():

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python all_cause_parameter_count.py "
            "<normalise_gbd_dataset>"
        )
        print()

        sys.exit(1)


    root = Path(
        sys.argv[1]
    )


    if not root.exists():

        print(
            f"Folder not found: "
            f"{root.resolve()}"
        )

        sys.exit(1)


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 90)
    print("ALL CAUSE - PARAMETER COUNT")
    print("=" * 90)

    print(
        f"Root folder : {root.resolve()}"
    )

    print(
        "Measures    : DALYs, Deaths, YLLs, YLDs"
    )

    print(
        "Metric      : Number only"
    )

    print()


    # --------------------------------------------------------
    # FIND ALL CSV FILES
    # --------------------------------------------------------

    files = list(
        root.rglob("*.csv")
    )


    if not files:

        print(
            "ERROR: No CSV files found."
        )

        sys.exit(1)


    print(
        f"CSV files found: {len(files):,}"
    )

    print()


    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {
        "cause_id",
        "cause_name",
        "metric_name",
    }


    # Dictionary:
    #
    # measure
    #     cause_id
    #         count
    #
    counts = {
        measure: {}
        for measure in MEASURES
    }


    # Cause names
    cause_names = {}


    scanned = 0
    used = 0
    skipped = 0


    # ========================================================
    # PROCESS FILES
    # ========================================================

    for file_path in files:

        scanned += 1


        # ----------------------------------------------------
        # Detect measure
        # ----------------------------------------------------

        path_measure = detect_measure_from_path(
            file_path,
            root
        )


        # ----------------------------------------------------
        # Read header
        # ----------------------------------------------------

        try:

            header = pd.read_csv(
                file_path,
                nrows=0
            )

        except Exception as exc:

            skipped += 1

            print(
                f"[SKIP] {file_path}"
            )

            print(
                f"       {exc}"
            )

            continue


        columns = set(
            header.columns
        )


        # ----------------------------------------------------
        # Check required columns
        # ----------------------------------------------------

        missing = (
            required_columns
            - columns
        )


        if missing:

            skipped += 1

            continue


        # ----------------------------------------------------
        # If measure is not found in path,
        # check measure_name column
        # ----------------------------------------------------

        has_measure_column = (
            "measure_name" in columns
        )


        if (
            path_measure is None
            and not has_measure_column
        ):

            skipped += 1

            continue


        # ----------------------------------------------------
        # Read required columns
        # ----------------------------------------------------

        usecols = [
            "cause_id",
            "cause_name",
            "metric_name",
        ]


        if has_measure_column:

            usecols.append(
                "measure_name"
            )


        try:

            df = pd.read_csv(
                file_path,
                usecols=usecols,
                low_memory=False
            )

        except Exception as exc:

            skipped += 1

            print(
                f"[SKIP] Could not read: "
                f"{file_path}"
            )

            print(
                f"       {exc}"
            )

            continue


        # ====================================================
        # DETERMINE MEASURE
        # ====================================================

        if path_measure is not None:

            df["measure"] = (
                path_measure
            )

        else:

            df["measure"] = (
                df["measure_name"]
                .map(
                    normalize_measure_name
                )
            )


        # Keep only four parameters
        df = df[
            df["measure"].isin(
                MEASURES
            )
        ]


        # ====================================================
        # METRIC = NUMBER ONLY
        # ====================================================

        df["metric_name"] = (
            df["metric_name"]
            .astype(str)
            .str.strip()
            .str.lower()
        )


        df = df[
            df["metric_name"]
            == METRIC.lower()
        ]


        if df.empty:

            continue


        # ====================================================
        # CLEAN CAUSE
        # ====================================================

        df["cause_id"] = pd.to_numeric(
            df["cause_id"],
            errors="coerce"
        )


        df["cause_name"] = (
            df["cause_name"]
            .astype(str)
            .str.strip()
        )


        df = df.dropna(
            subset=[
                "cause_id",
                "cause_name",
            ]
        )


        df = df[
            (df["cause_name"] != "")
            &
            (
                df["cause_name"]
                .str.lower()
                != "nan"
            )
        ]


        if df.empty:

            continue


        df["cause_id"] = (
            df["cause_id"]
            .astype(int)
        )


        # ====================================================
        # COUNT RECORDS
        # ====================================================

        grouped = (
            df.groupby(
                [
                    "measure",
                    "cause_id",
                    "cause_name",
                ],
                as_index=False
            )
            .size()
            .rename(
                columns={
                    "size": "count"
                }
            )
        )


        # ====================================================
        # STORE COUNTS
        # ====================================================

        for row in grouped.itertuples(
            index=False
        ):

            measure = str(
                row.measure
            )

            cause_id = int(
                row.cause_id
            )

            cause_name = str(
                row.cause_name
            )

            count = int(
                row.count
            )


            # Store cause name
            cause_names[
                cause_id
            ] = cause_name


            # Add count
            counts[
                measure
            ][
                cause_id
            ] = (
                counts[
                    measure
                ].get(
                    cause_id,
                    0
                )
                +
                count
            )


        used += 1


    # ========================================================
    # CREATE FINAL TABLE
    # ========================================================

    print()
    print(
        f"CSV files scanned : {scanned:,}"
    )

    print(
        f"CSV files used    : {used:,}"
    )

    print(
        f"CSV files skipped : {skipped:,}"
    )

    print()


    # All causes across all four parameters
    all_cause_ids = set(
        cause_names.keys()
    )


    for measure in MEASURES:

        all_cause_ids.update(
            counts[
                measure
            ].keys()
        )


    rows = []


    for cause_id in sorted(
        all_cause_ids
    ):

        row = {

            "Cause ID":
                cause_id,

            "Cause Name":
                cause_names.get(
                    cause_id,
                    ""
                ),

            "DALYs":
                counts[
                    "DALYs"
                ].get(
                    cause_id,
                    0
                ),

            "Deaths":
                counts[
                    "Deaths"
                ].get(
                    cause_id,
                    0
                ),

            "YLLs":
                counts[
                    "YLLs"
                ].get(
                    cause_id,
                    0
                ),

            "YLDs":
                counts[
                    "YLDs"
                ].get(
                    cause_id,
                    0
                ),
        }


        rows.append(
            row
        )


    final = pd.DataFrame(
        rows,
        columns=[
            "Cause ID",
            "Cause Name",
            "DALYs",
            "Deaths",
            "YLLs",
            "YLDs",
        ]
    )


    # ========================================================
    # SORT BY CAUSE ID
    # ========================================================

    final = final.sort_values(
        "Cause ID"
    ).reset_index(
        drop=True
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    final.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("=" * 90)
    print("COMPLETED")
    print("=" * 90)

    print(
        f"Unique causes : "
        f"{len(final):,}"
    )

    print(
        f"DALYs records : "
        f"{final['DALYs'].sum():,}"
    )

    print(
        f"Deaths records: "
        f"{final['Deaths'].sum():,}"
    )

    print(
        f"YLLs records  : "
        f"{final['YLLs'].sum():,}"
    )

    print(
        f"YLDs records  : "
        f"{final['YLDs'].sum():,}"
    )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    print()

    print(
        "Sample:"
    )

    print(
        final.head(10).to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()