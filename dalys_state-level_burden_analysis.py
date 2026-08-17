import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# SECTION 11 — STATE-WISE + YEAR-WISE DALYs TREND ANALYSIS
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
#   Indian states
#
# Cause:
#   All available causes
#
# Metric:
#   Number
#
# OUTPUT:
#   1 final CSV
#   1 PNG heatmap per REI
#
# Ranking:
#   1 = highest DALYs burden
#   Ranking is calculated separately for:
#       REI + Year
#
# Final CSV row sorting:
#   REI ID + START_YEAR Rank ascending
#   This makes the first displayed rank column:
#       1, 2, 3, 4, ...
#
# Display:
#   Million -> M
#   Thousand -> K
#
# Example:
#   5,215,476 -> 5.22M
#   521,547   -> 521.55K
#
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

METRIC = "Number"

START_YEAR = 2013
END_YEAR = 2023

OUTPUT_DIR = Path("section_11_output")

OUTPUT_FILE = (
    OUTPUT_DIR
    / "section_11_statewise_dalys_trend.csv"
)

HEATMAP_DIR = OUTPUT_DIR / "heatmaps"


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

    print("=" * 90)
    print("SECTION 11 — STATE-WISE + YEAR-WISE DALYs TREND ANALYSIS")
    print("=" * 90)
    print()
    print(f"Root folder : {root.resolve()}")
    print(f"DALYs files : {len(files)}")
    print(f"Metric      : {METRIC}")
    print(f"Years       : {START_YEAR}-{END_YEAR}")
    print("Age         : ALL")
    print("Sex         : Male + Female")
    print("Location    : ALL STATES")
    print("Cause       : ALL")
    print("REIs        : 20 COMMON REIs")
    print()

    all_data = []

    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"[SKIP] Could not read: {file}")
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
            print("[SKIP] Missing columns:")
            print(file)
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
            df["metric_name"].str.lower()
            == METRIC.lower()
        ]

        # ----------------------------------------------------
        # YEAR
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
        # SEX
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

        # ----------------------------------------------------
        # VALUE
        # ----------------------------------------------------
        df["val"] = pd.to_numeric(
            df["val"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["val"]
        )

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------
        df["location_name"] = (
            df["location_name"]
            .astype(str)
            .str.strip()
        )

        if df.empty:
            continue

        all_data.append(df)

    if not all_data:
        print()
        print("ERROR: No DALYs data found.")
        sys.exit(1)

    combined = pd.concat(
        all_data,
        ignore_index=True
    )

    print(f"Rows loaded: {len(combined):,}")

    return combined


# ============================================================
# CALCULATE STATE + YEAR DALYs
# ============================================================

def calculate_state_year_values(df):
    print()
    print("=" * 90)
    print("CALCULATING STATE-WISE ANNUAL DALYs")
    print("=" * 90)

    # Sum across Male + Female, all age groups and all causes.
    # Keep State + REI + Year.
    state_year = (
        df.groupby(
            [
                "location_name",
                "rei_id",
                "rei_name",
                "year",
            ],
            as_index=False
        )["val"]
        .sum()
    )

    state_year = state_year.sort_values(
        [
            "rei_id",
            "year",
            "location_name",
        ]
    )

    return state_year


# ============================================================
# CALCULATE STATE RANK
# ============================================================

def calculate_rank(state_year):
    print()
    print("=" * 90)
    print("CALCULATING STATE RANKINGS")
    print("=" * 90)

    state_year = state_year.copy()

    # 1 = highest DALYs burden.
    # Ranking is independent for each REI + year.
    state_year["rank"] = (
        state_year
        .groupby(
            [
                "rei_id",
                "year",
            ]
        )["val"]
        .rank(
            method="min",
            ascending=False
        )
        .astype("Int64")
    )

    return state_year


# ============================================================
# CALCULATE YoY
# ============================================================

def calculate_yoy(state_year):
    print()
    print("=" * 90)
    print("CALCULATING STATE-WISE YoY")
    print("=" * 90)

    state_year = state_year.copy()

    # IMPORTANT:
    # Sort once before shift so previous value/year are aligned.
    state_year = state_year.sort_values(
        [
            "location_name",
            "rei_id",
            "rei_name",
            "year",
        ]
    ).reset_index(drop=True)

    group_keys = [
        "location_name",
        "rei_id",
        "rei_name",
    ]

    state_year["previous_year_val"] = (
        state_year
        .groupby(group_keys)["val"]
        .shift(1)
    )

    state_year["previous_year"] = (
        state_year
        .groupby(group_keys)["year"]
        .shift(1)
    )

    # Calculate YoY only when the previous observation is exactly
    # the immediately preceding year and denominator is non-zero.
    valid_previous_year = (
        state_year["previous_year"]
        == state_year["year"] - 1
    )

    denominator_valid = (
        state_year["previous_year_val"].notna()
        & (state_year["previous_year_val"] != 0)
    )

    valid_yoy = (
        valid_previous_year
        & denominator_valid
    )

    state_year["yoy_pct_change"] = pd.NA

    state_year.loc[
        valid_yoy,
        "yoy_pct_change"
    ] = (
        (
            state_year.loc[valid_yoy, "val"]
            - state_year.loc[valid_yoy, "previous_year_val"]
        )
        / state_year.loc[valid_yoy, "previous_year_val"]
        * 100
    )

    return state_year


# ============================================================
# FORMAT DALYs
# ============================================================

def format_dalys(value):
    if pd.isna(value):
        return ""

    value = float(value)
    absolute = abs(value)

    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"{value / 1_000:.2f}K"

    return f"{value:.2f}"


# ============================================================
# FORMAT YoY
# ============================================================

def format_yoy(value):
    if pd.isna(value):
        return ""

    value = float(value)

    if value > 0:
        return f"+{value:.2f}%"

    return f"{value:.2f}%"


# ============================================================
# CREATE FINAL WIDE TABLE
# ============================================================

def create_final_table(state_year):
    print()
    print("=" * 90)
    print("CREATING FINAL STATE-WISE WIDE TABLE")
    print("=" * 90)

    rows = []

    for (
        location_name,
        rei_id,
        rei_name
    ), group in state_year.groupby(
        [
            "location_name",
            "rei_id",
            "rei_name",
        ]
    ):
        group = group.sort_values("year")

        row = {
            "State": location_name,
            "REI ID": int(rei_id),
            "REI": rei_name,
        }

        lookup = {
            int(item["year"]): item
            for _, item in group.iterrows()
        }

        for year in range(
            START_YEAR,
            END_YEAR + 1
        ):
            current = lookup.get(year)

            if current is not None:
                row[f"{year} DALYs"] = format_dalys(
                    current["val"]
                )

                row[f"{year} Rank"] = (
                    int(current["rank"])
                    if not pd.isna(current["rank"])
                    else ""
                )
            else:
                row[f"{year} DALYs"] = ""
                row[f"{year} Rank"] = ""

            if year > START_YEAR:
                if current is not None:
                    row[
                        f"{year - 1}-{year} YoY %"
                    ] = format_yoy(
                        current["yoy_pct_change"]
                    )
                else:
                    row[
                        f"{year - 1}-{year} YoY %"
                    ] = ""

        rows.append(row)

    final = pd.DataFrame(rows)

    # --------------------------------------------------------
    # COLUMN ORDER
    # --------------------------------------------------------
    columns = [
        "State",
        "REI ID",
        "REI",
    ]

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):
        columns.append(f"{year} DALYs")
        columns.append(f"{year} Rank")

        if year > START_YEAR:
            columns.append(
                f"{year - 1}-{year} YoY %"
            )

    final = final[columns]

    # --------------------------------------------------------
    # SORTING FIX
    # --------------------------------------------------------
    # Previously this table was sorted by 2023 Rank. Therefore
    # the 2013 Rank column looked random when inspected.
    #
    # Because this is one wide row per State + REI, the row order
    # can only follow one year's rank at a time. We use START_YEAR
    # so 2013 Rank is displayed in ascending order: 1, 2, 3, ...
    sort_rank_column = f"{START_YEAR} Rank"

    final["_sort_rank"] = pd.to_numeric(
        final[sort_rank_column],
        errors="coerce"
    )

    final = final.sort_values(
        [
            "REI ID",
            "_sort_rank",
            "State",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        na_position="last"
    )

    final = final.drop(
        columns=["_sort_rank"]
    )

    final = final.reset_index(drop=True)

    return final


# ============================================================
# HEATMAP HELPERS
# ============================================================

def safe_filename(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "rei"


def create_heatmaps(state_year):
    print()
    print("=" * 90)
    print("CREATING DALYs HEATMAP IMAGES")
    print("=" * 90)

    HEATMAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    heatmap_files = []

    years = list(range(START_YEAR, END_YEAR + 1))

    for rei_id, rei_group in state_year.groupby("rei_id"):
        rei_group = rei_group.copy()

        rei_names = (
            rei_group["rei_name"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        rei_name = (
            rei_names.iloc[0]
            if not rei_names.empty
            else f"REI {int(rei_id)}"
        )

        # ----------------------------------------------------
        # Sort states by START_YEAR rank so heatmap row order
        # matches the corrected CSV ordering for each REI.
        # ----------------------------------------------------
        start_year_ranks = (
            rei_group[
                rei_group["year"] == START_YEAR
            ][
                ["location_name", "rank"]
            ]
            .drop_duplicates(
                subset=["location_name"]
            )
            .rename(
                columns={"rank": "start_rank"}
            )
        )

        pivot = rei_group.pivot_table(
            index="location_name",
            columns="year",
            values="val",
            aggfunc="sum"
        )

        # Ensure all requested years appear in the heatmap.
        pivot = pivot.reindex(columns=years)

        # Attach 2013 rank for sorting rows.
        pivot = (
            pivot
            .reset_index()
            .merge(
                start_year_ranks,
                on="location_name",
                how="left"
            )
            .sort_values(
                ["start_rank", "location_name"],
                ascending=[True, True],
                na_position="last"
            )
            .set_index("location_name")
        )

        pivot = pivot.drop(
            columns=["start_rank"],
            errors="ignore"
        )

        if pivot.empty:
            continue

        # Show values in thousands or millions depending on the
        # largest absolute value, keeping the heatmap readable.
        max_abs = pivot.abs().max().max()

        if pd.isna(max_abs):
            continue

        if max_abs >= 1_000_000:
            display_matrix = pivot / 1_000_000
            unit_label = "DALYs (millions)"
        elif max_abs >= 1_000:
            display_matrix = pivot / 1_000
            unit_label = "DALYs (thousands)"
        else:
            display_matrix = pivot.copy()
            unit_label = "DALYs"

        state_count = len(display_matrix.index)

        figure_height = max(
            7.0,
            min(18.0, state_count * 0.38 + 2.5)
        )

        fig, ax = plt.subplots(
            figsize=(15, figure_height)
        )

        image = ax.imshow(
            display_matrix.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest"
        )

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, ha="right")

        ax.set_yticks(range(state_count))
        ax.set_yticklabels(display_matrix.index.tolist())

        ax.set_xlabel("Year")
        ax.set_ylabel("State")
        ax.set_title(
            f"State-wise DALYs Heatmap — {rei_name} "
            f"(REI {int(rei_id)}), {START_YEAR}-{END_YEAR}"
        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            pad=0.02
        )
        colorbar.set_label(unit_label)

        fig.tight_layout()

        filename = (
            f"rei_{int(rei_id)}_"
            f"{safe_filename(rei_name)}_"
            f"dalys_heatmap.png"
        )

        output_path = HEATMAP_DIR / filename

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        heatmap_files.append(output_path)

        print(f"Saved heatmap: {output_path}")

    return heatmap_files


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print(
            "python section_11_statewise_trend.py "
            "<normalised_gbd_dataset>"
        )
        sys.exit(1)

    root = Path(sys.argv[1])

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

    # STEP 1 — LOAD
    df = load_dalys(root)

    # STEP 2 — STATE + YEAR VALUES
    state_year = calculate_state_year_values(df)

    # STEP 3 — RANKING
    state_year = calculate_rank(state_year)

    # STEP 4 — YoY
    state_year = calculate_yoy(state_year)

    # STEP 5 — FINAL WIDE TABLE
    final = create_final_table(state_year)

    # STEP 6 — SAVE CSV
    final.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # STEP 7 — SAVE HEATMAP IMAGES
    heatmap_files = create_heatmaps(state_year)

    # SUMMARY
    print()
    print("=" * 90)
    print("SECTION 11 COMPLETE")
    print("=" * 90)
    print()
    print(
        f"Common REIs analysed : "
        f"{final['REI ID'].nunique()}"
    )
    print(
        f"States analysed      : "
        f"{final['State'].nunique()}"
    )
    print(
        f"Years analysed       : "
        f"{START_YEAR}-{END_YEAR}"
    )
    print(
        f"Final rows           : "
        f"{len(final):,}"
    )
    print(
        f"Heatmaps created     : "
        f"{len(heatmap_files)}"
    )
    print()
    print("Ranking:")
    print("1 = Highest DALYs burden")
    print(
        "Ranking is calculated separately "
        "for each REI and year."
    )
    print(
        f"CSV rows are sorted by {START_YEAR} Rank "
        "ascending within each REI."
    )
    print()
    print("DALYs normalisation:")
    print("1,000,000+ -> M")
    print("1,000+     -> K")
    print()
    print("CSV output:")
    print(OUTPUT_FILE.resolve())
    print()
    print("Heatmap folder:")
    print(HEATMAP_DIR.resolve())
    print()
    print(
        final.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()