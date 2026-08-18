import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 14 - RISK CATEGORY ANALYSIS
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python risk_category_analysis.py <input_folder>")
    sys.exit(1)


# ============================================================
# 1. INPUT / OUTPUT
# ============================================================

INPUT_DIR = sys.argv[1]

DALYS_DIR = os.path.join(
    INPUT_DIR,
    "DALYs"
)

OUTPUT_DIR = "section_14_output"

IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "image"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# 2. COMMON 20 REIs
# ============================================================

COMMON_REIS = {
    91: "Lead exposure",
    99: "Smoking",
    100: "Secondhand smoke",
    102: "High alcohol use",
    105: "High fasting plasma glucose",
    107: "High systolic blood pressure",
    108: "High body-mass index",
    116: "Diet high in red meat",
    117: "Diet high in processed meat",
    118: "Diet high in sugar-sweetened beverages",
    119: "Diet low in fiber",
    121: "Diet low in seafood omega-3 fatty acids",
    122: "Diet low in omega-6 polyunsaturated fatty acids",
    123: "Diet high in trans fatty acids",
    124: "Diet high in sodium",
    125: "Low physical activity",
    332: "Chewing tobacco",
    341: "Kidney dysfunction",
    367: "High LDL cholesterol",
    380: "Particulate matter pollution"
}


# ============================================================
# 3. RISK CATEGORY MAPPING
# ============================================================

RISK_CATEGORY = {

    # Environmental / Occupational
    91: "Environmental/Occupational",
    380: "Environmental/Occupational",

    # Metabolic
    105: "Metabolic",
    107: "Metabolic",
    108: "Metabolic",
    341: "Metabolic",
    367: "Metabolic",

    # Behavioral
    99: "Behavioral",
    100: "Behavioral",
    102: "Behavioral",
    116: "Behavioral",
    117: "Behavioral",
    118: "Behavioral",
    119: "Behavioral",
    121: "Behavioral",
    122: "Behavioral",
    123: "Behavioral",
    124: "Behavioral",
    125: "Behavioral",
    332: "Behavioral"
}


CATEGORY_ORDER = [
    "Behavioral",
    "Environmental/Occupational",
    "Metabolic"
]


YEARS = list(
    range(2013, 2024)
)


# ============================================================
# 4. CHECK CATEGORY MAPPING
# ============================================================

missing_category = [
    rei_id
    for rei_id in COMMON_REIS
    if rei_id not in RISK_CATEGORY
]

if missing_category:

    print(
        "ERROR: Risk category missing for REI IDs:"
    )

    print(
        missing_category
    )

    sys.exit(1)


# ============================================================
# 5. FIND DALYs NUMBER FILES ONLY
# ============================================================

print(
    "\nSearching DALYs Number CSV files..."
)

csv_files = []

for root, dirs, files in os.walk(
    DALYS_DIR
):

    for file in files:

        if (
            file.lower().endswith(".csv")
            and "DALYs - Number -" in file
        ):

            csv_files.append(
                os.path.join(
                    root,
                    file
                )
            )


if not csv_files:

    print(
        "ERROR: No DALYs Number CSV files found."
    )

    sys.exit(1)


print(
    f"Found {len(csv_files)} DALYs Number files."
)


# ============================================================
# 6. LOAD DATA
# ============================================================

dataframes = []

for file_path in csv_files:

    try:

        temp_df = pd.read_csv(
            file_path,
            low_memory=False
        )

        dataframes.append(
            temp_df
        )

        print(
            "Loaded:",
            os.path.relpath(
                file_path,
                DALYS_DIR
            )
        )

    except Exception as e:

        print(
            f"Skipped: {file_path}"
        )

        print(
            f"Reason: {e}"
        )


if not dataframes:

    print(
        "ERROR: No valid CSV files loaded."
    )

    sys.exit(1)


df = pd.concat(
    dataframes,
    ignore_index=True
)


print(
    f"\nTotal rows loaded: {len(df):,}"
)


# ============================================================
# 7. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "rei_id",
    "year",
    "metric_name",
    "val"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print(
        "\nERROR: Missing required columns:"
    )

    for column in missing_columns:

        print(
            "-",
            column
        )

    print(
        "\nAvailable columns:"
    )

    print(
        list(df.columns)
    )

    sys.exit(1)


# ============================================================
# 8. CLEAN DATA
# ============================================================

df["rei_id"] = pd.to_numeric(
    df["rei_id"],
    errors="coerce"
)

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["val"] = pd.to_numeric(
    df["val"],
    errors="coerce"
)

df["metric_name"] = (
    df["metric_name"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# 9. METRIC = NUMBER ONLY
# ============================================================

df = df[
    df["metric_name"] == "number"
].copy()

print(
    f"After Metric = Number: {len(df):,}"
)


# ============================================================
# 10. COMMON 20 REIs ONLY
# ============================================================

df = df[
    df["rei_id"].isin(
        COMMON_REIS.keys()
    )
].copy()

print(
    f"After Common REI filter: {len(df):,}"
)


# ============================================================
# 11. YEAR 2013–2023
# ============================================================

df = df[
    df["year"].isin(
        YEARS
    )
].copy()

print(
    f"After Year 2013-2023 filter: {len(df):,}"
)


# ============================================================
# 12. REMOVE INVALID VALUES
# ============================================================

df = df.dropna(
    subset=[
        "rei_id",
        "year",
        "val"
    ]
)


if df.empty:

    print(
        "\nERROR: No data remaining after filtering."
    )

    sys.exit(1)


# ============================================================
# 13. ASSIGN RISK CATEGORY
# ============================================================

df["Risk Category"] = (
    df["rei_id"]
    .map(RISK_CATEGORY)
)


if df["Risk Category"].isna().any():

    missing_ids = (
        df.loc[
            df["Risk Category"].isna(),
            "rei_id"
        ]
        .unique()
        .tolist()
    )

    print(
        "ERROR: Missing category for REIs:"
    )

    print(
        missing_ids
    )

    sys.exit(1)


# ============================================================
# 14. YEAR + CATEGORY BURDEN
#
# Category Burden = SUM(Value)
#
# Aggregates:
# - all states
# - male + female
# - all age groups
# - all available causes
# ============================================================

category_year = (
    df.groupby(
        [
            "year",
            "Risk Category"
        ],
        as_index=False
    )["val"]
    .sum()
    .rename(
        columns={
            "year": "Year",
            "val": "DALYs Category Burden"
        }
    )
)


# ============================================================
# 15. ENSURE ALL YEAR + CATEGORY COMBINATIONS
# ============================================================

full_index = pd.MultiIndex.from_product(
    [
        YEARS,
        CATEGORY_ORDER
    ],
    names=[
        "Year",
        "Risk Category"
    ]
)


category_year = (
    category_year
    .set_index(
        [
            "Year",
            "Risk Category"
        ]
    )
    .reindex(
        full_index,
        fill_value=0
    )
    .reset_index()
)


# ============================================================
# 16. TOTAL BURDEN FOR EACH YEAR
# ============================================================

category_year["Total Year Burden"] = (
    category_year
    .groupby(
        "Year"
    )["DALYs Category Burden"]
    .transform("sum")
)


# ============================================================
# 17. CONTRIBUTION %
# ============================================================

category_year[
    "DALYs Contribution %"
] = np.where(

    category_year[
        "Total Year Burden"
    ] != 0,

    (
        category_year[
            "DALYs Category Burden"
        ]
        /
        category_year[
            "Total Year Burden"
        ]
        *
        100
    ),

    0
)


category_year[
    "DALYs Contribution %"
] = (
    category_year[
        "DALYs Contribution %"
    ]
    .round(2)
)


# ============================================================
# 18. CREATE FINAL 3-ROW TABLE
#
# Each year has two columns:
#
# 2013 DALYs (M)
# 2013 Contribution %
#
# ...
#
# 2023 DALYs (M)
# 2023 Contribution %
# ============================================================

final_result = pd.DataFrame(
    {
        "Risk Category": CATEGORY_ORDER
    }
)


for year in YEARS:

    year_data = category_year[
        category_year["Year"] == year
    ].set_index(
        "Risk Category"
    )


    # --------------------------------------------------------
    # Convert DALYs to Millions
    # --------------------------------------------------------

    final_result[
        f"{year} DALYs (M)"
    ] = (

        final_result[
            "Risk Category"
        ]
        .map(
            year_data[
                "DALYs Category Burden"
            ]
        )
        .fillna(0)
        / 1_000_000

    ).round(2)


    # --------------------------------------------------------
    # Contribution %
    # --------------------------------------------------------

    final_result[
        f"{year} Contribution %"
    ] = (

        final_result[
            "Risk Category"
        ]
        .map(
            year_data[
                "DALYs Contribution %"
            ]
        )
        .fillna(0)

    ).round(2)


# ============================================================
# 19. VALIDATE EACH YEAR
# ============================================================

print(
    "\nYear-wise contribution validation:"
)

for year in YEARS:

    total_percentage = (
        final_result[
            f"{year} Contribution %"
        ]
        .sum()
    )

    print(
        f"{year}: "
        f"{total_percentage:.2f}%"
    )

    if not np.isclose(
        total_percentage,
        100,
        atol=0.1
    ):

        print(
            f"WARNING: {year} does not total 100%."
        )


# ============================================================
# 20. SAVE CSV
# ============================================================

csv_output = os.path.join(
    OUTPUT_DIR,
    "step14_risk_category_analysis.csv"
)


final_result.to_csv(
    csv_output,
    index=False
)


# 22. BAR CHART
#
# DALYs in MILLIONS
# ============================================================

bar_pivot = category_year.pivot(
    index="Year",
    columns="Risk Category",
    values="DALYs Category Burden"
)


# Convert to millions
bar_pivot = (
    bar_pivot
    / 1_000_000
)


bar_pivot = bar_pivot[
    CATEGORY_ORDER
]


ax = bar_pivot.plot(
    kind="bar",
    figsize=(16, 9)
)


ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "DALYs Category Burden (Millions)"
)

ax.set_title(
    "Year-wise DALYs Burden by Risk Category"
)

ax.tick_params(
    axis="x",
    rotation=45
)

ax.legend(
    title="Risk Category"
)


plt.tight_layout()


bar_image = os.path.join(
    IMAGE_DIR,
    "step14_category_burden_yearwise_bar.png"
)


plt.savefig(
    bar_image,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 23. PIE CHART FOR EACH YEAR
# ============================================================

for year in YEARS:

    year_data = category_year[
        category_year["Year"] == year
    ].copy()


    # Convert burden to millions
    values_million = (
        year_data[
            "DALYs Category Burden"
        ]
        / 1_000_000
    )


    plt.figure(
        figsize=(8, 8)
    )


    plt.pie(
        values_million,
        labels=year_data[
            "Risk Category"
        ],
        autopct="%1.1f%%",
        startangle=90
    )


    plt.title(
        f"Risk Category DALYs Contribution - {year}"
    )


    plt.tight_layout()


    pie_image = os.path.join(
        IMAGE_DIR,
        f"step14_category_contribution_pie_{year}.png"
    )


    plt.savefig(
        pie_image,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Pie chart saved: {pie_image}"
    )


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("STEP 14 COMPLETED")
print("=" * 75)

print(
    f"CSV   : {csv_output}"
)

print(
    f"Images: {IMAGE_DIR}"
)

print(
    f"Rows  : {len(final_result)}"
)

print(
    f"Years : {len(YEARS)}"
)

print("\nFinal table:")

print(
    final_result.to_string(
        index=False
    )
)

print("\nDone.")