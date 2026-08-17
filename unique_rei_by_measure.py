import sys
from pathlib import Path
import pandas as pd


MEASURES = ["Deaths", "DALYs", "YLLs", "YLDs"]


def get_measure_from_path(csv_path, root):
    """
    Identify the GBD measure from the folder structure.
    """
    relative_parts = csv_path.relative_to(root).parts

    measure_map = {
        "deaths": "Deaths",
        "dalys": "DALYs",
        "ylls": "YLLs",
        "ylds": "YLDs",
    }

    for part in relative_parts:
        key = part.strip().lower()

        if key in measure_map:
            return measure_map[key]

    return None


def scan_files(root):
    """
    Recursively scan all CSV files and collect unique
    (rei_id, rei_name) pairs for each measure.
    """

    results = {
        measure: set()
        for measure in MEASURES
    }

    csv_files = list(root.rglob("*.csv"))

    print("=" * 80)
    print("GBD UNIQUE REI ID / NAME SCANNER")
    print("=" * 80)
    print(f"Root folder : {root.resolve()}")
    print(f"CSV files   : {len(csv_files)}")
    print()

    skipped = []

    for csv_file in csv_files:

        measure = get_measure_from_path(
            csv_file,
            root
        )

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

        df = (
            df[
                ["rei_id", "rei_name"]
            ]
            .dropna()
            .drop_duplicates()
        )

        for _, row in df.iterrows():

            try:
                rei_id = int(float(row["rei_id"]))
            except (ValueError, TypeError):
                continue

            rei_name = str(
                row["rei_name"]
            ).strip()

            results[measure].add(
                (rei_id, rei_name)
            )

    return results, skipped


def make_dataframe(values):
    """
    Convert unique REI pairs into a DataFrame.
    """

    if not values:
        return pd.DataFrame(
            columns=[
                "rei_id",
                "rei_name"
            ]
        )

    df = pd.DataFrame(
        sorted(values),
        columns=[
            "rei_id",
            "rei_name"
        ]
    )

    return df


def check_consistency(df, measure):
    """
    Check:
      1. One rei_id -> multiple names
      2. One rei_name -> multiple IDs
    """

    if df.empty:
        return

    # --------------------------------------------------------
    # Same rei_id -> multiple rei_name
    # --------------------------------------------------------

    id_check = (
        df.groupby("rei_id")["rei_name"]
        .nunique()
    )

    bad_ids = id_check[
        id_check > 1
    ]

    if not bad_ids.empty:

        print()
        print(
            f"[{measure}] WARNING: "
            "rei_id mapped to multiple rei_name values"
        )

        for rei_id in bad_ids.index:

            names = sorted(
                df.loc[
                    df["rei_id"] == rei_id,
                    "rei_name"
                ].unique()
            )

            print(
                f"  {rei_id}: {names}"
            )

    # --------------------------------------------------------
    # Same rei_name -> multiple rei_id
    # --------------------------------------------------------

    name_check = (
        df.groupby("rei_name")["rei_id"]
        .nunique()
    )

    bad_names = name_check[
        name_check > 1
    ]

    if not bad_names.empty:

        print()
        print(
            f"[{measure}] WARNING: "
            "rei_name mapped to multiple rei_id values"
        )

        for name in bad_names.index:

            ids = sorted(
                df.loc[
                    df["rei_name"] == name,
                    "rei_id"
                ].unique()
            )

            print(
                f"  {name}: {ids}"
            )


def create_comparison(dataframes):
    """
    Create master comparison table.

    Columns:
        rei_id
        rei_name
        Deaths
        DALYs
        YLLs
        YLDs
        measure_count
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

        for measure in MEASURES:

            df = dataframes[measure]

            if df.empty:

                row[measure] = False

            else:

                exists = (
                    (
                        df["rei_id"] == rei_id
                    )
                    &
                    (
                        df["rei_name"] == rei_name
                    )
                ).any()

                row[measure] = bool(
                    exists
                )

        row["measure_count"] = sum(
            row[measure]
            for measure in MEASURES
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    # ========================================================
    # COMMAND LINE ARGUMENT
    # ========================================================

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python unique_rei_by_measure.py "
            "<folder>"
        )
        print()
        print("Example:")
        print(
            r"python unique_rei_by_measure.py "
            r".\normalise_gbd_dataset\"
        )

        sys.exit(1)

    root = Path(
        sys.argv[1]
    )

    if not root.exists():

        print()
        print(
            f"ERROR: Folder does not exist:\n"
            f"{root.resolve()}"
        )

        sys.exit(1)

    if not root.is_dir():

        print()
        print(
            f"ERROR: Path is not a folder:\n"
            f"{root.resolve()}"
        )

        sys.exit(1)

    # ========================================================
    # SCAN
    # ========================================================

    results, skipped = scan_files(
        root
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = Path(
        "rei_analysis"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    dataframes = {}

    # ========================================================
    # INDIVIDUAL MEASURE RESULTS
    # ========================================================

    for measure in MEASURES:

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

    for measure in MEASURES:

        print(
            f"{measure:<10}: "
            f"{len(dataframes[measure])}"
        )

    print()
    print(
        f"Total unique REI pairs: "
        f"{len(comparison)}"
    )

    # ========================================================
    # MISSING REI BY MEASURE
    # ========================================================

    print()
    print("=" * 80)
    print("MISSING REI BY MEASURE")
    print("=" * 80)

    for measure in MEASURES:

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
    # PRESENT IN ALL FOUR
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
    # NOT PRESENT IN ALL FOUR
    # ========================================================

    not_all_four = comparison[
        comparison["measure_count"] < 4
    ]

    print()
    print("=" * 80)
    print("REI NOT PRESENT IN ALL FOUR MEASURES")
    print("=" * 80)

    print(
        f"Count: {len(not_all_four)}"
    )

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
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # SKIPPED FILES
    # ========================================================

    if skipped:

        print()
        print("=" * 80)
        print(
            "FILES WITHOUT A RECOGNIZED "
            "MEASURE FOLDER"
        )
        print("=" * 80)

        for file in skipped:
            print(file)

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print()
    print(
        f"Results saved in:\n"
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()