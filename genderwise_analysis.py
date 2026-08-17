import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 12
# SEX-WISE + CAUSE-WISE + YEAR-WISE + STATE-WISE ANALYSIS
# ============================================================


# ============================================================
# 1. INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python genderwise_analysis.py <input_folder>")
    sys.exit(1)

INPUT_DIR = sys.argv[1]

DALYS_DIR = os.path.join(INPUT_DIR, "DALYs")


# ============================================================
# 2. OUTPUT
# ============================================================

OUTPUT_DIR = "section_12_output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "image")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# 3. COMMON 20 REIs ONLY
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
# 4. FIND DALYs NUMBER FILES ONLY
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
                os.path.join(root, file)
            )


if not csv_files:

    print(
        f"ERROR: No DALYs Number files found in {DALYS_DIR}"
    )

    sys.exit(1)


print(
    f"Found {len(csv_files)} DALYs Number files."
)


# ============================================================
# 5. LOAD DATA
# ============================================================

dataframes = []

for file_path in csv_files:

    try:

        temp = pd.read_csv(
            file_path,
            low_memory=False
        )

        dataframes.append(temp)

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

    print("ERROR: No valid files loaded.")
    sys.exit(1)


df = pd.concat(
    dataframes,
    ignore_index=True
)


print(
    f"\nTotal rows loaded: {len(df):,}"
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "rei_id",
    "rei_name",
    "location_name",
    "sex_name",
    "year",
    "metric_name",
    "val"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    print("\nERROR: Missing columns:")

    for col in missing_columns:
        print("-", col)

    print("\nAvailable columns:")
    print(list(df.columns))

    sys.exit(1)


# ============================================================
# 7. CLEAN COLUMNS
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

df["sex_name"] = (
    df["sex_name"]
    .astype(str)
    .str.strip()
)

df["location_name"] = (
    df["location_name"]
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
# 8. METRIC = NUMBER ONLY
# ============================================================

df = df[
    df["metric_name"] == "number"
].copy()

print(
    f"After Metric = Number: {len(df):,}"
)


# ============================================================
# 9. COMMON 20 REIs ONLY
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
# 10. YEAR 2013–2023
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
# 11. MALE + FEMALE ONLY
# ============================================================

df["sex_clean"] = (
    df["sex_name"]
    .str.casefold()
)

df = df[
    df["sex_clean"].isin(
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
# 12. REMOVE INVALID VALUES
# ============================================================

df = df.dropna(
    subset=[
        "rei_id",
        "year",
        "location_name",
        "val"
    ]
)


if df.empty:

    print(
        "\nERROR: No data remaining after filtering."
    )

    sys.exit(1)


# ============================================================
# 13. USE COMMON REI NAMES
# ============================================================

df["rei_name"] = df["rei_id"].map(
    COMMON_REIS
)


# ============================================================
# 14. STATE + YEAR + REI + SEX
#
# Burden = SUM(Value)
# ============================================================

summary = (
    df.groupby(
        [
            "location_name",
            "year",
            "rei_id",
            "rei_name",
            "sex_clean"
        ],
        as_index=False
    )["val"]
    .sum()
    .rename(
        columns={
            "val": "Burden"
        }
    )
)


# ============================================================
# 15. PIVOT MALE / FEMALE
# ============================================================

result = summary.pivot_table(
    index=[
        "location_name",
        "year",
        "rei_id",
        "rei_name"
    ],
    columns="sex_clean",
    values="Burden",
    aggfunc="sum",
    fill_value=0
).reset_index()


# ============================================================
# 16. ENSURE BOTH SEXES EXIST
# ============================================================

if "male" not in result.columns:
    result["male"] = 0

if "female" not in result.columns:
    result["female"] = 0


# ============================================================
# 17. RENAME COLUMNS
# ============================================================

result = result.rename(
    columns={
        "location_name": "State",
        "year": "Year",
        "rei_id": "REI ID",
        "rei_name": "REI Name",
        "male": "Male Burden",
        "female": "Female Burden"
    }
)


# ============================================================
# 18. STATE BURDEN
#
# State Burden =
# Male Burden + Female Burden
# ============================================================

result["State Burden"] = (
    result["Male Burden"]
    +
    result["Female Burden"]
)


# ============================================================
# 19. M/F RATIO
#
# M/F Ratio =
# Male Burden / Female Burden
# ============================================================

result["M/F Ratio"] = np.where(
    result["Female Burden"] != 0,
    result["Male Burden"]
    /
    result["Female Burden"],
    np.nan
)


# ============================================================
# 20. STATE RANK
#
# Rank states for each:
#
# REI + Year
#
# Highest State Burden = Rank 1
# ============================================================

result["State Rank"] = (
    result
    .groupby(
        [
            "REI ID",
            "Year"
        ]
    )["State Burden"]
    .rank(
        method="min",
        ascending=False
    )
)


result["State Rank"] = (
    result["State Rank"]
    .astype(int)
)


# ============================================================
# 21. ROUND RATIO
# ============================================================

result["M/F Ratio"] = (
    result["M/F Ratio"]
    .round(2)
)


# ============================================================
# 22. SORT
#
# REI → Year → State Rank
# ============================================================

result = result.sort_values(
    [
        "REI ID",
        "Year",
        "State Rank"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 23. FINAL COLUMN ORDER
# ============================================================

result = result[
    [
        "REI ID",
        "REI Name",
        "Year",
        "State Rank",
        "State",
        "Male Burden",
        "Female Burden",
        "State Burden",
        "M/F Ratio"
    ]
]


# ============================================================
# 24. SAVE CSV
# ============================================================

csv_output = os.path.join(
    OUTPUT_DIR,
    "step12_cause_year_state_ranking.csv"
)


result.to_csv(
    csv_output,
    index=False
)


# ============================================================
# 25. GRAPH
#
# State ranking for each REI can be very large,
# so create a heatmap-like ranking plot using
# the top states for each REI in 2023.
# ============================================================

plot_year = 2023

plot_df = result[
    result["Year"] == plot_year
].copy()


# Take top 5 states for every REI
plot_df = (
    plot_df
    .sort_values(
        [
            "REI ID",
            "State Rank"
        ]
    )
    .groupby(
        "REI Name"
    )
    .head(5)
)


# Create figure
plt.figure(
    figsize=(18, 12)
)


# Plot each REI's top 5 states
for rei_name, group in plot_df.groupby(
    "REI Name"
):

    plt.plot(
        group["State Rank"],
        [rei_name] * len(group),
        "o"
    )


plt.xlabel(
    "State Rank"
)

plt.ylabel(
    "Risk Factor"
)

plt.title(
    "Top State Rankings by Risk Factor - 2023"
)

plt.tight_layout()


# ============================================================
# 26. SAVE GRAPH
# ============================================================

image_output = os.path.join(
    IMAGE_DIR,
    "step12_state_ranking_2023.png"
)


plt.savefig(
    image_output,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 27. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("STEP 12 COMPLETED")
print("=" * 70)

print(
    f"CSV          : {csv_output}"
)

print(
    f"Image        : {image_output}"
)

print(
    f"Rows         : {len(result):,}"
)

print(
    f"REIs         : {result['REI ID'].nunique()}"
)

print(
    f"States       : {result['State'].nunique()}"
)

print(
    f"Years        : {result['Year'].nunique()}"
)

print("\nFirst 20 rows:")

print(
    result.head(20).to_string(
        index=False
    )
)

print("\nDone.")