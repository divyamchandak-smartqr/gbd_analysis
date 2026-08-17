import sys
from pathlib import Path
import pandas as pd


# ============================================================
# USAGE
# ============================================================
# python unique_rei_by_measure.py <folder>
#
# Example:
# python unique_rei_by_measure.py .\normalise_gbd_dataset\
#
# The folder can contain any number of nested subfolders.
# The script recursively finds every CSV.
# ============================================================


def get_measure_from_path(csv_path, root):
    """
    Determine the GBD measure from the folder structure.

    Expected structure somewhere below root:

        Deaths/
        DALYs/
        YLLs/
        YLDs/

    Example:
        normalise_gbd_dataset/Deaths/Behaviour/file.csv
    """

    relative_parts = csv_path.relative_to(root).parts

    measures = {
        "deaths": "Deaths",
        "dalys": "DALYs",
        "ylls": "YLLs",
        "ylds": "YLDs",
    }

    for part in relative_parts:
        key = part.strip().lower()

        if key in measures:
            return measures[key]

    return None


def scan_files(root):
    """
    Recursively scan all CSV files and extract unique
    rei_id / rei_name combinations by measure.
    """

    results = {
        "Deaths": set(),
        "DALYs": set(),
        "YLLs": set(),
        "YLDs": set(),
    }

    all_csv_files = list(root.rglob("*.csv"))

    print("=" * 80)
    print("GBD UNIQUE REI ID / NAME SCANNER")
    print("=" * 80)
    print(f"Root folder : {root.resolve()}")
    print(f"CSV files   : {len(all_csv_files)}")
    print()

    skipped = []

    for csv_file in all_csv_files:

        measure = get_measure_from_path(csv_file, root)

        if measure is None:
            skipped.append(csv_file)
            continue

        try:
            df = pd.read_csv(
                csv_file,
                usecols=["rei_id", "rei_name"]
            )

        except ValueError:
            print(
                f"[SKIP - MISSING REI COLUMNS] "
                f"{csv_file}"
            )
            continue

        except Exception as e:
            print(
                f"[ERROR] {csv_file}\n"
                f"        {e}"
            )
            continue

        df = df[
            ["rei_id", "rei_name"]
        ].dropna().drop_duplicates()

        for _, row in df.iterrows():

            try:
                rei_id = int(float(row["rei_id"]))
            except (ValueError, TypeError):
                continue

            rei_name = str(row["rei_name"]).strip()

            results[measure].add(
                (rei_id, rei_name)
            )

    return results, skipped


def make_dataframe(values):
    """
    Convert set of (rei_id, rei_name) into DataFrame.
    """

    if not values:
        return pd.DataFrame(
            columns=["rei_id", "rei_name"]
        )

    df = pd.DataFrame(
        sorted(values),
        columns=["rei_id", "rei_name"]
    )

    return df


def check_consistency(df, measure):
    """
    Check whether the same rei_id appears with
    different names or the same name appears with
    different IDs.
    """

    if df.empty:
        return

    # --------------------------------------------------------
    # Same ID -> multiple names
    # --------------------------------------------------------

    id_check = (
        df.groupby("rei_id")["rei_name"]
        .nunique()
    )

    bad_ids = id_check[id_check > 1]

    if not bad_ids.empty:

        print(
            f"\n[{measure}] WARNING: "
            "rei_id mapped to multiple rei_name values"
        )

        for rei_id in bad_ids.index:

            names = sorted(
                df.loc[
                    df["rei_id"] == rei_id,
                    "rei_name"
                ].unique()
            )

            print(f"  {rei_id}: {names}")

    # --------------------------------------------------------
    # Same name -> multiple IDs
    # --------------------------------------------------------

    name_check = (
        df.groupby("rei_name")["rei_id"]
        .nunique()
    )

    bad_names = name_check[name_check > 1]

    if not bad_names.empty:

        print(
            f"\n[{measure}] WARNING: "
            "rei_name mapped to multiple rei_id values"
        )

        for name in bad_names.index:

            ids = sorted(
                df.loc[
                    df["rei_name"] == name,
                    "rei_id"
                ].unique()
            )

            print(f"  {name}: {ids}")


def create_comparison(dataframes):
    """
    Create one master table showing whether every REI
    exists in each measure.
    """

    all_pairs = set()

    for df in dataframes.values():

        if not df.empty:

            all_pairs.update(
                zip(
                    df["rei_id"],
                    df["rei_name"]
                )
            )

    rows = []

    for rei_id, rei_name in sorted(all_pairs):

        row = {
            "rei_id": rei_id,
            "rei_name": rei_name,
        }

        for measure in [
            "Deaths",
            "DALYs",
            "YLLs",
            "YLDs"
        ]:

            if dataframes[measure].empty:

                row[measure] = False

            else:

                row[measure] = (
                    (
                        dataframes[measure]["rei_id"]
                        == rei_id
                    )
                    &
                    (
                        dataframes[measure]["rei_name"]
                        == rei_name
                    )
                ).any()

        row["measure_count"] = sum(
            row[m]
            for m in [
                "Deaths",
                "DALYs",
                "YLLs",
                "YLDs"
            ]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    # ========================================================
    # CHECK ARGUMENT
    # ========================================================

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "    python unique_rei_by_measure.py <folder>\n\n"
            "Example:\n"
            "    python unique_rei_by_measure.py "
            ".\\normalise_gbd_dataset\\"
        )

        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():

        print(
            f"\nERROR: Folder does not exist:\n"
            f"{root.resolve()}"
        )

        sys.exit(1)

    if not root.is_dir():

        print(
            f"\nERROR: Path is not a folder:\n"
            f"{root.resolve()}"
        )

        sys.exit(1)

    # ========================================================
    # SCAN
    # ========================================================

    results, skipped = scan_files(root)

    output_dir = Path("rei_analysis")
    output_dir.mkdir(exist_ok=True)

    # ========================================================
    # CREATE INDIVIDUAL MEASURE TABLES
    # ========================================================

    dataframes = {}

    for measure in [
        "Deaths",
        "DALYs",
        "YLLs",
        "YLDs"
    ]:

        df = make_dataframe(
            results[measure]
        )

        dataframes[measure] = df

        check_consistency(
            df,
            measure
        )

        output_file = (
            output_dir
            / f"unique_rei_{measure.lower()}.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print()
        print("=" * 80)
        print(measure)
        print("=" * 80)
        print(
            f"Unique rei_id / rei_name pairs: "
            f"{len(df)}"
        )
        print(
            f"Output: {output_file}"
        )

        if not df.empty:
            print()
            print(
                df.to_string(
                    index=False
                )
            )

    # ========================================================
    # CROSS-MEASURE COMPARISON
    # ========================================================

    comparison = create_comparison(
        dataframes
    )

    comparison_file = (
        output_dir
        / "unique_rei_all_measures.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False
    )

    print()
    print("=" * 80)
    print("CROSS-MEASURE REI COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for measure in [
        "Deaths",
        "DALYs",
        "YLLs",
        "YLDs"
    ]:

        print(
            f"{measure:<10}: "
            f"{len(dataframes[measure])}"
        )

    print(
        f"\nTotal unique REI pairs: "
        f"{len(comparison)}"
    )

    # ========================================================
    # MEASURE COVERAGE
    # ========================================================

    print()
    print("=" * 80)
    print("MISSING REI BY MEASURE")
    print("=" * 80)

    for measure in [
        "Deaths",
        "DALYs",
        "YLLs",
        "YLDs"
    ]:

        missing = comparison[
            comparison[measure] == False
        ]

        print()
        print(
            f"--- {measure}: "
            f"{len(missing)} not present ---"
        )

        if not missing.empty:

            print(
                missing[
                    [
                        "rei_id",
                        "rei_name"
                    ]
                ].to_string(
                    index=False
                )
            )

    # ========================================================
    # REI PRESENT IN ALL FOUR
    # ========================================================

    all_four = comparison[
        comparison["measure_count"] == 4
    ]

    print()
    print("=" * 80)
    print("REI PRESENT IN ALL FOUR MEASURES")
    print("=" * 80)

    print(
        f"Count: {len(all_four)}"
    )

    if not all_four.empty:

        print(
            all_four[
                [
                    "rei_id",
                    "rei_name"
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # REI NOT PRESENT IN ALL FOUR
    # ========================================================

    not_all_four = comparison[
        comparison["measure_count"] < 4
    ]

    print()
    print("=" * 80)
    print("REI NOT PRESENT IN ALL FOUR MEASURES")
    print("=" * 80)

    if not not_all_four.empty:

        print(
            not_all_four[
                [
                    "rei_id",
                    "rei_name",
                    "Deaths",
                    "DALYs",
                    "YLLs",
                    "YLDs",
                    "measure_count"
                ].__str__()
                if False
                else not_all_four[
                    [
                        "rei_id",
                        "rei_name",
                        "Deaths",
                        "DALYs",
                        "YLLs",
                        "YLDs",
                        "measure_count"
                    ]
                ].to_string(index=False)
        )

    # ========================================================
    # SKIPPED FILES
    # ========================================================

    if skipped:

        print()
        print("=" * 80)
        print("FILES WITHOUT A RECOGNIZED MEASURE FOLDER")
        print("=" * 80)

        for file in skipped:
            print(file)

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"\nResults saved in:\n"
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()