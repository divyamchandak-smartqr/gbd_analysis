import sys
from pathlib import Path
import pandas as pd


MEASURES = ["Deaths", "DALYs", "YLLs", "YLDs"]


def get_measure(csv_file, root):
    """
    Identify Deaths / DALYs / YLLs / YLDs
    from the folder path.
    """

    parts = csv_file.relative_to(root).parts

    for part in parts:
        name = part.strip().lower()

        if name == "deaths":
            return "Deaths"
        elif name == "dalys":
            return "DALYs"
        elif name == "ylls":
            return "YLLs"
        elif name == "ylds":
            return "YLDs"

    return None


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python unique_common_rei.py <folder>\n\n"
            "Example:\n"
            "python unique_common_rei.py .\\normalise_gbd_dataset\\"
        )
        sys.exit(1)

    root = Path(sys.argv[1])

    if not root.exists():
        print(f"Folder not found: {root}")
        sys.exit(1)

    # --------------------------------------------------------
    # Store unique REI pairs for each measure
    # --------------------------------------------------------

    rei_by_measure = {
        measure: set()
        for measure in MEASURES
    }

    csv_files = list(root.rglob("*.csv"))

    print("=" * 70)
    print("SCANNING GBD DATASET")
    print("=" * 70)
    print(f"Root   : {root.resolve()}")
    print(f"CSV files found: {len(csv_files)}")

    # --------------------------------------------------------
    # Read every CSV recursively
    # --------------------------------------------------------

    for csv_file in csv_files:

        measure = get_measure(
            csv_file,
            root
        )

        if measure is None:
            continue

        try:

            df = pd.read_csv(
                csv_file,
                usecols=[
                    "rei_id",
                    "rei_name"
                ]
            )

        except Exception as e:

            print(
                f"[SKIP] {csv_file}\n"
                f"       {e}"
            )

            continue

        df = (
            df[
                [
                    "rei_id",
                    "rei_name"
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

        for _, row in df.iterrows():

            try:
                rei_id = int(
                    float(row["rei_id"])
                )
            except:
                continue

            rei_name = str(
                row["rei_name"]
            ).strip()

            rei_by_measure[measure].add(
                (
                    rei_id,
                    rei_name
                )
            )

    # --------------------------------------------------------
    # Print unique REIs for each measure
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UNIQUE REI BY MEASURE")
    print("=" * 70)

    for measure in MEASURES:

        print()
        print(
            f"{measure}: "
            f"{len(rei_by_measure[measure])} unique REIs"
        )

        for rei_id, rei_name in sorted(
            rei_by_measure[measure]
        ):

            print(
                f"{rei_id:<5} {rei_name}"
            )

    # --------------------------------------------------------
    # Find common REIs
    # --------------------------------------------------------

    common_rei = (
        rei_by_measure["Deaths"]
        & rei_by_measure["DALYs"]
        & rei_by_measure["YLLs"]
        & rei_by_measure["YLDs"]
    )

    # --------------------------------------------------------
    # Print common REIs
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMMON REI IN ALL FOUR MEASURES")
    print("=" * 70)

    print(
        f"Total common REIs: {len(common_rei)}"
    )

    print()

    for rei_id, rei_name in sorted(
        common_rei
    ):

        print(
            f"{rei_id:<5} {rei_name}"
        )

    # --------------------------------------------------------
    # Save common REIs
    # --------------------------------------------------------

    output_dir = Path(
        "rei_analysis"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir
        / "common_rei_all_four_measures.csv"
    )

    common_df = pd.DataFrame(
        sorted(
            common_rei
        ),
        columns=[
            "rei_id",
            "rei_name"
        ]
    )

    common_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Common REIs saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()