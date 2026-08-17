import sys
from pathlib import Path
import pandas as pd


# ============================================================
# UNIQUE REI BY MEASURE + COMMON REI ACROSS ALL 4 MEASURES
# ============================================================

MEASURES = [
    "Deaths",
    "DALYs",
    "YLLs",
    "YLDs",
]


def find_measure(file_path, root):
    """
    Determine measure from the folder structure.
    """
    parts = file_path.relative_to(root).parts

    for part in parts:
        if part.lower() in {
            "deaths",
            "dalys",
            "ylls",
            "ylds",
        }:
            return part

    return None


def load_rei_data(root):

    data = {
        measure: {}
        for measure in MEASURES
    }

    csv_files = list(
        root.rglob("*.csv")
    )

    print(
        f"Total CSV files found: {len(csv_files)}"
    )

    for csv_file in csv_files:

        measure = find_measure(
            csv_file,
            root
        )

        if measure is None:
            continue

        # Normalize measure name
        measure_key = measure.upper()

        if measure_key not in {
            "DEATHS",
            "DALYS",
            "YLLS",
            "YLDS",
        }:
            continue

        measure_name = {
            "DEATHS": "Deaths",
            "DALYS": "DALYs",
            "YLLS": "YLLs",
            "YLDS": "YLDs",
        }[measure_key]

        try:
            df = pd.read_csv(
                csv_file,
                usecols=[
                    "rei_id",
                    "rei_name",
                ],
            )

        except Exception as e:

            print(
                f"[ERROR] {csv_file}"
            )

            print(e)

            continue

        df = df.dropna(
            subset=[
                "rei_id",
                "rei_name",
            ]
        )

        for _, row in df.iterrows():

            try:
                rei_id = int(
                    float(row["rei_id"])
                )
            except Exception:
                continue

            rei_name = str(
                row["rei_name"]
            ).strip()

            if not rei_name:
                continue

            data[
                measure_name
            ][rei_id] = rei_name

    return data


def main():

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )
        print(
            "python unique_rei_common.py "
            "<folder>"
        )
        print()
        print(
            "Example:"
        )
        print()

        sys.exit(1)

    root = Path(
        sys.argv[1]
    )

    if not root.exists():

        print(
            f"Folder not found: {root.resolve()}"
        )

        sys.exit(1)

    print()
    print("=" * 80)
    print("UNIQUE REI ANALYSIS")
    print("=" * 80)

    print(
        f"Root folder: {root.resolve()}"
    )

    data = load_rei_data(
        root
    )

    # ========================================================
    # UNIQUE REI PER MEASURE
    # ========================================================

    print()
    print("=" * 80)
    print("UNIQUE REI PER MEASURE")
    print("=" * 80)

    for measure in MEASURES:

        print()
        print(
            f"{measure}: "
            f"{len(data[measure])} unique REIs"
        )

        for rei_id in sorted(
            data[measure]
        ):

            print(
                f"{rei_id:<5} "
                f"{data[measure][rei_id]}"
            )

    # ========================================================
    # COMMON TO ALL FOUR
    # ========================================================

    sets = {
        measure: set(
            data[measure].keys()
        )
        for measure in MEASURES
    }

    common_ids = (
        sets["Deaths"]
        & sets["DALYs"]
        & sets["YLLs"]
        & sets["YLDs"]
    )

    print()
    print("=" * 80)
    print("COMMON REI IDs IN ALL 4 MEASURES")
    print("=" * 80)

    print(
        f"Common REIs: {len(common_ids)}"
    )

    for rei_id in sorted(
        common_ids
    ):

        print(
            f"{rei_id:<5} "
            f"{data['Deaths'][rei_id]}"
        )

    # ========================================================
    # COMMON TABLE
    # ========================================================

    rows = []

    for rei_id in sorted(
        common_ids
    ):

        rows.append(
            {
                "rei_id": rei_id,
                "rei_name": data[
                    "Deaths"
                ][rei_id],
            }
        )

    common_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = Path(
        "rei_analysis"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    common_file = (
        output_dir
        / "common_rei_all_4_measures.csv"
    )

    common_df.to_csv(
        common_file,
        index=False
    )

    # ========================================================
    # SAVE ALL UNIQUE REIs
    # ========================================================

    all_rows = []

    all_ids = set()

    for measure in MEASURES:
        all_ids.update(
            data[measure].keys()
        )

    for rei_id in sorted(
        all_ids
    ):

        all_rows.append(
            {
                "rei_id": rei_id,
                "rei_name": (
                    data["Deaths"].get(rei_id)
                    or data["DALYs"].get(rei_id)
                    or data["YLLs"].get(rei_id)
                    or data["YLDs"].get(rei_id)
                ),
                "Deaths": (
                    "YES"
                    if rei_id in data["Deaths"]
                    else "NO"
                ),
                "DALYs": (
                    "YES"
                    if rei_id in data["DALYs"]
                    else "NO"
                ),
                "YLLs": (
                    "YES"
                    if rei_id in data["YLLs"]
                    else "NO"
                ),
                "YLDs": (
                    "YES"
                    if rei_id in data["YLDs"]
                    else "NO"
                ),
            }
        )

    all_df = pd.DataFrame(
        all_rows
    )

    all_file = (
        output_dir
        / "all_unique_rei_by_measure.csv"
    )

    all_df.to_csv(
        all_file,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        f"All unique REIs : "
        f"{len(all_df)}"
    )

    print(
        f"Common in all 4 : "
        f"{len(common_df)}"
    )

    print()

    print(
        f"Saved:"
    )

    print(
        common_file.resolve()
    )

    print(
        all_file.resolve()
    )


if __name__ == "__main__":
    main()