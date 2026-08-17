"""
extract_rei_reference.py

Scans every CSV under a root folder and builds a reference table of every
unique (rei_id, rei_name) risk-factor pair found in the data.

Also flags two kinds of problems that matter for this pipeline:
  1. ID/NAME CONSISTENCY — same rei_id showing up under different rei_name
     text, or the same rei_name text showing up under different rei_id.
     Either of these means two "different" risk factors in your files are
     actually the same thing (or vice versa) and any group-by on rei_name
     alone would silently mis-group them.
  2. AGGREGATE VS COMPONENT OVERLAP — checks whether a known aggregate risk
     (e.g. "Particulate matter pollution", rei_id 380) and its component
     sub-risks (e.g. "Ambient particulate matter pollution" id 86,
     "Household air pollution from solid fuels" id 87) ever appear together
     in the SAME measure+category+age file. If they do, summing that file's
     rows blindly will double-count. Reports which measures/categories use
     the aggregate vs. the disaggregated version so you can pick one level
     and standardize.

Output:
  - rei_reference.csv: rid, rname, list of (measure, category) combos it appears in
  - printed report of any consistency or overlap issues found

Usage:
    python extract_rei_reference.py <root_dir> [output_csv]
"""

import sys
import csv
import glob
from pathlib import Path
from collections import defaultdict

# risk factors known (or suspected) to be an aggregate of others, based on
# GBD's risk hierarchy. Edit this if you confirm/deny the relationship via
# GBD's own hierarchy docs. Format: aggregate_id -> set of component_ids
KNOWN_AGGREGATE_CANDIDATES = {
    380: {86, 87},  # "Particulate matter pollution" vs Ambient + Household
}


def find_measure_category(path: Path):
    """Best-effort guess of measure/category from the path, tolerant of the
    raw export's inconsistent casing/folder naming. Used only for reporting
    context, not for correctness-critical logic."""
    parts_lower = [p.lower() for p in path.parts]
    measure = next((p for p in path.parts if p.lower() in
                     ("dalys", "deaths", "ylls", "ylds")), "UNKNOWN_MEASURE")
    category = "UNKNOWN_CATEGORY"
    for p in path.parts:
        pl = p.lower().strip()
        if pl.startswith("metabolic"):
            category = "Metabolic"
        elif pl.startswith("behav"):
            category = "Behavioural"
        elif pl.startswith("env"):
            category = "Environmental"
    # YLLs has no category subfolder — pull it from the filename instead
    if category == "UNKNOWN_CATEGORY":
        stem_lower = path.stem.lower()
        if "metabolic" in stem_lower:
            category = "Metabolic"
        elif "behav" in stem_lower:
            category = "Behavioural"
        elif "environmental" in stem_lower:
            category = "Environmental"
    return measure, category


def main(root_dir, out_csv="rei_reference.csv"):
    root = Path(root_dir)
    csv_files = sorted(root.rglob("*.csv"))
    print(f"Scanning {len(csv_files)} CSV files under {root}...")

    id_to_names = defaultdict(set)
    name_to_ids = defaultdict(set)
    # (rei_id, rei_name) -> set of (measure, category) it was found in
    pair_contexts = defaultdict(set)
    # (measure, category) -> set of rei_ids seen in that slice, for overlap check
    context_ids = defaultdict(set)

    for path in csv_files:
        measure, category = find_measure_category(path)
        try:
            with open(path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid_raw = row.get("rei_id")
                    rname = row.get("rei_name")
                    if rid_raw is None or rname is None:
                        continue
                    try:
                        rid = int(float(rid_raw))
                    except ValueError:
                        continue
                    id_to_names[rid].add(rname)
                    name_to_ids[rname].add(rid)
                    pair_contexts[(rid, rname)].add((measure, category))
                    context_ids[(measure, category)].add(rid)
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")

    # ---- write reference table --------------------------------------
    all_pairs = sorted(pair_contexts.keys(), key=lambda x: x[0])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rei_id", "rei_name", "measure_category_combos"])
        for rid, rname in all_pairs:
            combos = "; ".join(f"{m}/{c}" for m, c in sorted(pair_contexts[(rid, rname)]))
            w.writerow([rid, rname, combos])

    print(f"\nTotal unique (rei_id, rei_name) pairs: {len(all_pairs)}")
    print(f"Reference table written to: {out_csv}\n")
    for rid, rname in all_pairs:
        print(f"  {rid}\t{rname}")

    # ---- consistency checks ------------------------------------------
    print("\n--- CHECK: rei_id mapped to multiple rei_name values ---")
    found = False
    for rid, names in id_to_names.items():
        if len(names) > 1:
            found = True
            print(f"  id {rid} -> {names}")
    if not found:
        print("  none — every rei_id has exactly one name")

    print("\n--- CHECK: rei_name mapped to multiple rei_id values ---")
    found = False
    for rname, ids in name_to_ids.items():
        if len(ids) > 1:
            found = True
            print(f"  '{rname}' -> {ids}")
    if not found:
        print("  none — every rei_name has exactly one id")

    # ---- aggregate/component overlap check ----------------------------
    print("\n--- CHECK: aggregate risk vs its components appearing in the SAME file slice (double-count risk) ---")
    any_overlap = False
    for agg_id, component_ids in KNOWN_AGGREGATE_CANDIDATES.items():
        agg_name = next((n for i, n in pair_contexts if i == agg_id), f"id {agg_id}")
        for (measure, category), ids_here in context_ids.items():
            has_agg = agg_id in ids_here
            comps_here = component_ids & ids_here
            if has_agg and comps_here:
                any_overlap = True
                print(f"  OVERLAP in {measure}/{category}: aggregate id {agg_id} "
                      f"AND component id(s) {comps_here} both present — "
                      f"do NOT sum this file's rei rows without excluding one side.")
    if not any_overlap:
        print("  no direct overlap found in same file — but check across measures:")

    print("\n--- Where the aggregate vs. components are used, by measure ---")
    for agg_id, component_ids in KNOWN_AGGREGATE_CANDIDATES.items():
        agg_name = next((n for i, n in pair_contexts if i == agg_id), f"id {agg_id}")
        print(f"\n  Aggregate: id {agg_id} '{agg_name}'")
        for (rid, rname), combos in pair_contexts.items():
            if rid == agg_id:
                for m, c in sorted(combos):
                    print(f"    used in: {m}/{c}")
        for comp_id in component_ids:
            comp_name = next((n for i, n in pair_contexts if i == comp_id), f"id {comp_id}")
            print(f"  Component: id {comp_id} '{comp_name}'")
            for (rid, rname), combos in pair_contexts.items():
                if rid == comp_id:
                    for m, c in sorted(combos):
                        print(f"    used in: {m}/{c}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python extract_rei_reference.py <root_dir> [output_csv]")
        sys.exit(1)
    root_dir = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) == 3 else "rei_reference.csv"
    main(root_dir, out_csv)