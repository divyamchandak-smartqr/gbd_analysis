import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 10 — ANNUAL TREND ANALYSIS
# ============================================================
#
# DATA:
#   DALYs only
#
# REIs:
#   20 common REIs
#
# Age:
#   ALL age groups
#
# Years:
#   2013–2023
#
# Sex:
#   Male + Female
#
# Location:
#   All Indian states
#
# Cause:
#   All available CVD causes
#
# Metric:
#   Number
#
# OUTPUT:
#
#   rei_id
#   rei_name
#   2013
#   2014
#   2013-2014 YoY %
#   2015
#   2014-2015 YoY %
#   ...
#   2023
#   2022-2023 YoY %
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

METRIC = "Number"

START_YEAR = 2013
END_YEAR = 2023

OUTPUT_DIR = Path(
    "section_10_output"
)


# ============================================================
# 20 COMMON REIs
# ============================================================

COMMON_REI_IDS = {
    91,     # Lead exposure
    99,     # Smoking
    100,    # Secondhand smoke
    102,    # High alcohol use
    105,    # High fasting plasma glucose
    107,    # High systolic blood pressure
    108,    # High body-mass index
    116,    # Diet high in red meat
    117,    # Diet high in processed meat
    118,    # Diet high in sugar-sweetened beverages
    119,    # Diet low in fiber
    121,    # Diet low in seafood omega-3 fatty acids
    122,    # Diet low in omega-6 polyunsaturated fatty acids
    123,    # Diet high in trans fatty acids
    124,    # Diet high in sodium
    125,    # Low physical activity
    332,    # Chewing tobacco
    341,    # Kidney dysfunction
    367,    # High LDL cholesterol
    380,    # Particulate matter pollution
}


# ============================================================
# FIND DALYs FILES
# ============================================================

def find_dalys_files(root):

    files = []

    for file in root.rglob("*.csv"):

        relative_parts = [
            part.lower()
            for part in file.relative_to(root).parts
        ]

        if "dalys" in relative_parts:
            files.append(file)

    return files


# ============================================================
# LOAD DALYs
# ============================================================

def load_dalys(root):

    files = find_dalys_files(root)

    print("=" * 80)
    print("SECTION 10 — ANNUAL TREND ANALYSIS")
    print("=" * 80)

    print()
    print(f"Root folder : {root.resolve()}")
    print(f"DALYs files : {len(files)}")
    print(f"Metric      : {METRIC}")
    print(f"Years       : {START_YEAR}-{END_YEAR}")
    print("Age         : ALL")
    print("Sex         : Male + Female")
    print("Location    : ALL")
    print("Cause       : ALL")
    print("REIs        : 20 COMMON REIs")
    print()

    all_data = []

    for file in files:

        try:

            df = pd.read_csv(file)

        except Exception as e:

            print(
                f"[SKIP] Could not read: {file}"
            )

            print(e)

            continue

        required_columns = [
            "location_name",
            "sex_name",
            "age_name",
            "cause_name",
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

            print(
                "[SKIP] Missing columns in:"
            )

            print(file)
            print("Missing:", missing)

            continue

        # ----------------------------------------------------
        # Metric
        # ----------------------------------------------------

        df["metric_name"] = (
            df["metric_name"]
            .astype(str)
            .str.strip()
        )

        df = df[
            df["metric_name"].str.lower()
            == METRIC.lower()
        ]

        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Sex
        #
        # Keep Male + Female.
        #
        # There is no requirement for a "Both" row.
        # ----------------------------------------------------

        df["sex_name"] = (
            df["sex_name"]
            .astype(str)
            .str.strip()
        )

        df = df[
            df["sex_name"].isin(
                [
                    "Male",
                    "Female",
                ]
            )
        ]

        # ----------------------------------------------------
        # REI
        # ----------------------------------------------------

        df["rei_id"] = pd.to_numeric(
            df["rei_id"],
            errors="coerce"
        )

        df = df[
            df["rei_id"].isin(
                COMMON_REI_IDS
            )
        ]

        if df.empty:
            continue

        # ----------------------------------------------------
        # Value
        # ----------------------------------------------------

        df["val"] = pd.to_numeric(
            df["val"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["val"]
        )

        all_data.append(df)

    if not all_data:

        print()
        print(
            "ERROR: No DALYs data found."
        )

        sys.exit(1)

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    print(
        f"Rows loaded: {len(combined):,}"
    )

    return combined


# ============================================================
# CALCULATE ANNUAL DALYs
# ============================================================

def calculate_annual_values(df):

    print()
    print("=" * 80)
    print("CALCULATING ANNUAL DALYs")
    print("=" * 80)

    # --------------------------------------------------------
    # Sum across:
    #
    #   ALL locations
    #   Male + Female
    #   ALL age groups
    #   ALL CVD causes
    #
    # Group only by:
    #
    #   REI
    #   Year
    # --------------------------------------------------------

    annual = (
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

    annual = annual.sort_values(
        [
            "rei_id",
            "year",
        ]
    )

    return annual


# ============================================================
# CALCULATE YoY
# ============================================================

def calculate_yoy(annual):

    print()
    print("=" * 80)
    print("CALCULATING YEAR-ON-YEAR CHANGE")
    print("=" * 80)

    annual = annual.copy()

    # --------------------------------------------------------
    # Previous year value
    # --------------------------------------------------------

    annual["previous_year_val"] = (
        annual
        .groupby(
            [
                "rei_id",
                "rei_name",
            ]
        )["val"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Previous year
    # --------------------------------------------------------

    annual["previous_year"] = (
        annual
        .groupby(
            [
                "rei_id",
                "rei_name",
            ]
        )["year"]
        .shift(1)
    )

    # --------------------------------------------------------
    # YoY formula
    #
    # ((Current - Previous) / Previous) * 100
    # --------------------------------------------------------

    annual["yoy_pct_change"] = (
        (
            annual["val"]
            - annual["previous_year_val"]
        )
        / annual["previous_year_val"]
        * 100
    )

    # --------------------------------------------------------
    # First year has no previous year
    # --------------------------------------------------------

    annual.loc[
        annual["previous_year_val"].isna(),
        "yoy_pct_change"
    ] = pd.NA

    # --------------------------------------------------------
    # Previous value = 0
    # Avoid division by zero
    # --------------------------------------------------------

    annual.loc[
        annual["previous_year_val"] == 0,
        "yoy_pct_change"
    ] = pd.NA

    return annual


# ============================================================
# CREATE FINAL WIDE TABLE
# ============================================================

def create_final_table(annual):

    """
    Final table format:

    rei_id | rei_name | 2013 | 2014 | 2013-2014 YoY %
           | 2015 | 2014-2015 YoY % | ...
           | 2023 | 2022-2023 YoY %

    2013 has no YoY column because there is
    no 2012 value in the analysis period.
    """

    annual = annual.copy()

    annual["year"] = (
        annual["year"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Get REI information
    # --------------------------------------------------------

    rei_info = (
        annual[
            [
                "rei_id",
                "rei_name",
            ]
        ]
        .drop_duplicates(
            subset=["rei_id"]
        )
        .sort_values(
            "rei_id"
        )
    )

    final = rei_info.copy()

    # --------------------------------------------------------
    # Add each year's value
    # and following year's YoY
    # --------------------------------------------------------

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        year_data = annual[
            annual["year"] == year
        ][
            [
                "rei_id",
                "val",
                "yoy_pct_change",
            ]
        ].copy()

        # ----------------------------------------------------
        # Current year value
        # ----------------------------------------------------

        value_column = str(year)

        year_data = year_data.rename(
            columns={
                "val": value_column
            }
        )

        final = final.merge(
            year_data[
                [
                    "rei_id",
                    value_column,
                ]
            ],
            on="rei_id",
            how="left"
        )

        # ----------------------------------------------------
        # YoY column
        #
        # Example:
        #
        # 2014 -> 2013-2014 YoY %
        # 2015 -> 2014-2015 YoY %
        # ...
        # 2023 -> 2022-2023 YoY %
        # ----------------------------------------------------

        if year > START_YEAR:

            yoy_column = (
                f"{year - 1}-{year} YoY %"
            )

            yoy_data = (
                annual[
                    annual["year"] == year
                ][
                    [
                        "rei_id",
                        "yoy_pct_change",
                    ]
                ]
                .rename(
                    columns={
                        "yoy_pct_change":
                        yoy_column
                    }
                )
            )

            final = final.merge(
                yoy_data[
                    [
                        "rei_id",
                        yoy_column,
                    ]
                ],
                on="rei_id",
                how="left"
            )

    # --------------------------------------------------------
    # Round annual values
    # --------------------------------------------------------

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        column = str(year)

        if column in final.columns:

            final[column] = (
                pd.to_numeric(
                    final[column],
                    errors="coerce"
                )
                .round(2)
            )

    # --------------------------------------------------------
    # Round YoY percentages
    # --------------------------------------------------------

    for year in range(
        START_YEAR + 1,
        END_YEAR + 1
    ):

        column = (
            f"{year - 1}-{year} YoY %"
        )

        if column in final.columns:

            final[column] = (
                pd.to_numeric(
                    final[column],
                    errors="coerce"
                )
                .round(2)
            )

    # --------------------------------------------------------
    # EXACT COLUMN ORDER
    # --------------------------------------------------------

    ordered_columns = [
        "rei_id",
        "rei_name",
    ]

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        # Annual value
        ordered_columns.append(
            str(year)
        )

        # YoY after the annual value
        if year > START_YEAR:

            ordered_columns.append(
                f"{year - 1}-{year} YoY %"
            )

    final = final[
        ordered_columns
    ]

    return final


# ============================================================
# CREATE LINE GRAPHS
# ============================================================

def create_graphs(annual):

    graph_dir = (
        OUTPUT_DIR
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "Creating line graphs..."
    )

    for rei_id, group in annual.groupby(
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
            f"DALYs Annual Trend — {rei_name}"
        )

        plt.xlabel(
            "Year"
        )

        plt.ylabel(
            "DALYs Number"
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
            character
            if character.isalnum()
            or character in " _-"
            else "_"
            for character in rei_name
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
        f"Graphs saved to: "
        f"{graph_dir.resolve()}"
    )


# ============================================================
# PRINT FORMATTED TABLE
# ============================================================

def print_final_table(final):

    print()
    print("=" * 80)
    print("SECTION 10 — FINAL ANNUAL TREND TABLE")
    print("=" * 80)
    print()

    display = final.copy()

    # --------------------------------------------------------
    # Format annual values with commas
    # --------------------------------------------------------

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        column = str(year)

        if column in display.columns:

            display[column] = display[column].apply(
                lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
            )

    # --------------------------------------------------------
    # Format YoY percentages
    #
    # Example:
    # +3.03%
    # -2.46%
    # --------------------------------------------------------

    for year in range(
        START_YEAR + 1,
        END_YEAR + 1
    ):

        column = (
            f"{year - 1}-{year} YoY %"
        )

        if column in display.columns:

            display[column] = display[column].apply(
                lambda x:
                f"{x:+.2f}%"
                if pd.notna(x)
                else ""
            )

    print(
        display.to_string(
            index=False
        )
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
            "<normalise_gbd_dataset_folder>"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python annual_trend_analysis.py "
            ".\\normalise_gbd_dataset\\"
        )

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

    if not root.is_dir():

        print(
            f"Path is not a folder: "
            f"{root.resolve()}"
        )

        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # STEP 1 — LOAD DALYs
    # ========================================================

    df = load_dalys(
        root
    )

    # ========================================================
    # STEP 2 — CALCULATE ANNUAL VALUES
    # ========================================================

    annual = calculate_annual_values(
        df
    )

    # ========================================================
    # STEP 3 — CALCULATE YoY
    # ========================================================

    annual = calculate_yoy(
        annual
    )

    # ========================================================
    # STEP 4 — CREATE WIDE TABLE
    # ========================================================

    final = create_final_table(
        annual
    )

    # ========================================================
    # STEP 5 — SAVE CSV
    # ========================================================

    output_csv = (
        OUTPUT_DIR
        / "section_10_dalys_annual_trend.csv"
    )

    final.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # STEP 6 — CREATE GRAPHS
    # ========================================================

    create_graphs(
        annual
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("SECTION 10 COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Common REIs analysed : "
        f"{final['rei_id'].nunique()}"
    )

    print(
        f"Years analysed       : "
        f"{START_YEAR}-{END_YEAR}"
    )

    print(
        f"Rows                 : "
        f"{len(final):,}"
    )

    print()

    # ========================================================
    # PRINT FINAL WIDE TABLE
    # ========================================================

    print_final_table(
        final
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "Output:"
    )

    print(
        output_csv.resolve()
    )

    print()
    print(
        "Graphs:"
    )

    print(
        (
            OUTPUT_DIR
            / "graphs"
        ).resolve()
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()