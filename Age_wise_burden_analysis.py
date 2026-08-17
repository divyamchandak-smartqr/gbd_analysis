import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 13 - AGE-WISE BURDEN ANALYSIS
# ============================================================


# ============================================================
# 1. INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python agewise_analysis.py <input_folder>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]

DALYS_DIR = os.path.join(
    INPUT_DIR,
    "DALYs"
)


# ============================================================
# 2. OUTPUT
# ============================================================

OUTPUT_DIR = "section_13_output"

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
# 3. COMMON 20 REIs
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
# 4. REQUESTED AGE GROUPS
# ============================================================

AGE_ORDER = [
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69"
]


# ============================================================
# 5. INDIAN STATES / UNION TERRITORIES
# ============================================================

INDIAN_STATES = {
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry"
}


# ============================================================
# 6. FIND DALYs NUMBER FILES ONLY
# ============================================================

print("\nSearching DALYs Number CSV files...")

csv_files = []

for root, dirs, files in os.walk(DALYS_DIR):

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
# 7. LOAD CSV FILES
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
# 8. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "rei_id",
    "location_name",
    "sex_name",
    "age_name",
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
# 9. CLEAN COLUMNS
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

df["location_name"] = (
    df["location_name"]
    .astype(str)
    .str.strip()
)

df["sex_name"] = (
    df["sex_name"]
    .astype(str)
    .str.strip()
)

df["age_name"] = (
    df["age_name"]
    .astype(str)
    .str.strip()
)

df["metric_name"] = (
    df["metric_name"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# 10. METRIC = NUMBER ONLY
# ============================================================

df = df[
    df["metric_name"] == "number"
].copy()


print(
    f"After Metric = Number: {len(df):,}"
)


# ============================================================
# 11. COMMON 20 REIs ONLY
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
# 12. YEAR 2013-2023
# ============================================================

df = df[
    df["year"].between(
        2013,
        2023
    )
].copy()


print(
    f"After Year filter: {len(df):,}"
)


# ============================================================
# 13. MALE + FEMALE
# ============================================================

df = df[
    df["sex_name"]
    .str.casefold()
    .isin(
        [
            "male",
            "female"
        ]
    )
].copy()


print(
    f"After Male/Female filter: {len(df):,}"
)


# ============================================================
# 14. INDIAN STATES ONLY
# ============================================================

df = df[
    df["location_name"].isin(
        INDIAN_STATES
    )
].copy()


print(
    f"After Indian State filter: {len(df):,}"
)

print(
    f"States found: {df['location_name'].nunique()}"
)


# ============================================================
# 15. CREATE AGE GROUP
#
# Converts:
#
# 20-24 → 20-29
# 25-29 → 20-29
#
# 30-34 → 30-39
# 35-39 → 30-39
#
# etc.
# ============================================================

def map_age_group(age_name):

    text = str(age_name).strip().lower()

    # Extract first age number
    match = re.search(
        r"(\d+)\s*[-–]\s*(\d+)",
        text
    )

    if not match:
        return None

    start_age = int(
        match.group(1)
    )

    end_age = int(
        match.group(2)
    )

    if start_age >= 20 and end_age <= 29:
        return "20-29"

    if start_age >= 30 and end_age <= 39:
        return "30-39"

    if start_age >= 40 and end_age <= 49:
        return "40-49"

    if start_age >= 50 and end_age <= 59:
        return "50-59"

    if start_age >= 60 and end_age <= 69:
        return "60-69"

    return None


df["Age Group"] = (
    df["age_name"]
    .apply(map_age_group)
)


# Keep requested age groups only
df = df[
    df["Age Group"].isin(
        AGE_ORDER
    )
].copy()


print(
    f"After Age 20-69 filter: {len(df):,}"
)


# ============================================================
# 16. REMOVE INVALID VALUES
# ============================================================

df = df.dropna(
    subset=[
        "rei_id",
        "year",
        "location_name",
        "sex_name",
        "Age Group",
        "val"
    ]
)


if df.empty:

    print(
        "\nERROR: No data remaining after filtering."
    )

    sys.exit(1)


# ============================================================
# 17. ASSIGN REI NAME
# ============================================================

df["REI Name"] = (
    df["rei_id"]
    .map(COMMON_REIS)
)


# ============================================================
# 18. AGE GROUP BURDEN
#
# Age Group Burden = SUM(Value)
#
# REI + Year + State + Sex + Age Group
# ============================================================

summary = (
    df.groupby(
        [
            "rei_id",
            "REI Name",
            "year",
            "location_name",
            "sex_name",
            "Age Group"
        ],
        as_index=False
    )["val"]
    .sum()
    .rename(
        columns={
            "val": "Age Group Burden"
        }
    )
)


# ============================================================
# 19. TOTAL BURDEN ACROSS 20-69
# ============================================================

summary["Total Age Burden"] = (
    summary
    .groupby(
        [
            "rei_id",
            "year",
            "location_name",
            "sex_name"
        ]
    )["Age Group Burden"]
    .transform("sum")
)


# ============================================================
# 20. AGE CONTRIBUTION %
#
# Age Contribution =
# Age Group Burden
# --------------------------- × 100
# Total Age Burden
# ============================================================

summary["Age Contribution (%)"] = np.where(

    summary["Total Age Burden"] != 0,

    (
        summary["Age Group Burden"]
        /
        summary["Total Age Burden"]
        *
        100
    ),

    0
)


summary["Age Contribution (%)"] = (
    summary["Age Contribution (%)"]
    .round(2)
)


# ============================================================
# 21. PIVOT AGE GROUPS INTO COLUMNS
# ============================================================

result = summary.pivot_table(

    index=[
        "rei_id",
        "REI Name",
        "year",
        "location_name",
        "sex_name"
    ],

    columns="Age Group",

    values="Age Contribution (%)",

    aggfunc="sum",

    fill_value=0

).reset_index()


# ============================================================
# 22. ENSURE ALL AGE GROUP COLUMNS
# ============================================================

for age_group in AGE_ORDER:

    if age_group not in result.columns:

        result[age_group] = 0


# ============================================================
# 23. RENAME COLUMNS
# ============================================================

result = result.rename(
    columns={
        "rei_id": "REI ID",
        "year": "Year",
        "location_name": "State",
        "sex_name": "Sex"
    }
)


# ============================================================
# 24. ROUND AGE CONTRIBUTIONS
# ============================================================

for age_group in AGE_ORDER:

    result[age_group] = (
        result[age_group]
        .round(2)
    )


# ============================================================
# 25. SORT
#
# REI → Year → State → Sex
# ============================================================

result = result.sort_values(

    [
        "REI ID",
        "Year",
        "State",
        "Sex"
    ]

).reset_index(
    drop=True
)


# ============================================================
# 26. FINAL COLUMN ORDER
# ============================================================

result = result[
    [
        "REI ID",
        "REI Name",
        "Year",
        "State",
        "Sex",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69"
    ]
]


# ============================================================
# 27. VALIDATE 100%
# ============================================================

result["Percentage Total"] = (
    result[
        AGE_ORDER
    ]
    .sum(axis=1)
    .round(2)
)


invalid_rows = result[
    ~np.isclose(
        result["Percentage Total"],
        100,
        atol=0.1
    )
]


if len(invalid_rows) > 0:

    print(
        f"\nWARNING: {len(invalid_rows)} rows "
        "do not total approximately 100%."
    )

else:

    print(
        "\nPercentage validation: PASSED"
    )


result = result.drop(
    columns=[
        "Percentage Total"
    ]
)


# ============================================================
# 28. SAVE CSV
# ============================================================

csv_output = os.path.join(
    OUTPUT_DIR,
    "step13_age_wise_burden.csv"
)


result.to_csv(
    csv_output,
    index=False
)


# ============================================================
# 29. SAVE EXCEL
#
# One sheet only
# ============================================================

# ============================================================
# 30. CREATE STACKED BAR CHART
#
# One graph for EACH YEAR.
#
# Each graph contains:
# - 20 REIs
# - All states
# - Male + Female
# ============================================================

print(
    "\nCreating stacked bar charts..."
)


for plot_year in range(
    2013,
    2024
):

    plot_df = result[
        result["Year"] == plot_year
    ].copy()


    if plot_df.empty:

        print(
            f"No data for {plot_year}"
        )

        continue


    # --------------------------------------------------------
    # Create label
    # --------------------------------------------------------

    plot_df["Label"] = (
        plot_df["REI Name"]
        + " | "
        + plot_df["State"]
        + " | "
        + plot_df["Sex"]
    )


    # --------------------------------------------------------
    # X positions
    # --------------------------------------------------------

    x = np.arange(
        len(plot_df)
    )


    # --------------------------------------------------------
    # Bottom of stacked bars
    # --------------------------------------------------------

    bottom = np.zeros(
        len(plot_df)
    )


    # --------------------------------------------------------
    # Dynamic figure size
    # --------------------------------------------------------

    figure_width = max(
        25,
        len(plot_df) * 0.20
    )


    plt.figure(
        figsize=(
            figure_width,
            12
        )
    )


    # --------------------------------------------------------
    # STACK AGE GROUPS
    # --------------------------------------------------------

    for age_group in AGE_ORDER:

        values = (
            plot_df[age_group]
            .to_numpy()
        )


        plt.bar(
            x,
            values,
            bottom=bottom,
            label=age_group
        )


        bottom = (
            bottom
            +
            values
        )


    # --------------------------------------------------------
    # X AXIS
    # --------------------------------------------------------

    plt.xticks(
        x,
        plot_df["Label"],
        rotation=90,
        ha="center",
        fontsize=5
    )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    plt.xlabel(
        "REI | State | Sex"
    )


    plt.ylabel(
        "Age Group Contribution (%)"
    )


    plt.title(
        f"Age-wise Burden Distribution - {plot_year}"
    )


    # Every bar = 100%
    plt.ylim(
        0,
        100
    )


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    plt.legend(
        title="Age Group",
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    image_output = os.path.join(
        IMAGE_DIR,
        f"step13_age_wise_stacked_bar_{plot_year}.png"
    )


    plt.savefig(
        image_output,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {image_output}"
    )


# ============================================================
# 31. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 13 COMPLETED")
print("=" * 70)

print(
    f"CSV    : {csv_output}"
)

print(
    f"Images : {IMAGE_DIR}"
)

print(
    f"Rows   : {len(result):,}"
)

print(
    f"REIs   : {result['REI ID'].nunique()}"
)

print(
    f"States : {result['State'].nunique()}"
)

print(
    f"Years  : {result['Year'].nunique()}"
)

print(
    f"Sexes  : {result['Sex'].unique().tolist()}"
)


print("\nSample output:")

print(
    result.head(10).to_string(
        index=False
    )
)

print("\nDone.")