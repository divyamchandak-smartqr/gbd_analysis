"""
normalize_and_resolve.py

Combines normalize_filenames.py + resolve_conflicts.py into one pipeline run:

  STEP 1 — normalize
    Fixes the mess in the raw GBD export folders:
      - inconsistent category casing/spelling (Behaviour / BEHAVIOUR / "Behavioural " / Environment / ENV / ENVIRONMENT)
      - inconsistent delimiters (space-dash vs underscore)
      - inconsistent measure folder casing (DEATHs vs Deaths)
      - age-band typos in filenames (e.g. "50-55" when it should be "50-54")
    Cross-checks filename age against the actual age_name column inside each
    CSV. Never overwrites: if two source files claim the same canonical
    output name, BOTH are written under a "[CONFLICT - from ...]" suffix
    instead — nothing is silently lost.

  STEP 2 — resolve
    For every "[CONFLICT - from ...]" file produced in step 1, opens it and
    checks whether its content's age_name matches the canonical age band it's
    competing for.
      - Exactly one match -> that file is promoted to the canonical name,
        the other is moved to a "_needs_repull" folder (its true age band is
        missing from your dataset and needs a fresh GBD pull).
      - Zero or multiple matches -> left alone, flagged for manual review.
        The script never guesses when the data itself is ambiguous.

Nothing is ever deleted. Conflicting files are always renamed or moved, never
silently dropped.

Usage:
    python normalize_and_resolve.py <raw_root_dir> <output_dir>
"""

import sys
import re
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# ---- canonical vocab -------------------------------------------------

MEASURE_MAP = {
    "dalys": "DALYs",
    "deaths": "Deaths",
    "ylls": "YLLs",
    "ylds": "YLDs",
}

CATEGORY_MAP = {
    "metabolic": "Metabolic",
    "behaviour": "Behavioural",
    "behavioural": "Behavioural",
    "behavioral": "Behavioural",
    "env": "Environmental",
    "environment": "Environmental",
    "environmental": "Environmental",
}

METRIC_MAP = {
    "number": "Number",
    "rate": "Rate",
    "percent": "Percent",
}

VALID_AGE_BANDS = {
    "20-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64", "65-69",
}

AGE_RE = re.compile(r"(\d{2}-\d{2})")
CONFLICT_RE = re.compile(r"^(.*) \[CONFLICT - from (.*)\](\.csv)$")


def canonicalize(token, mapping):
    return mapping.get(token.strip().lower())


def parse_filename(path: Path):
    """Pull Measure / Metric / Category / Age out of a messy filename or its
    parent folders, regardless of delimiter or casing."""
    stem = path.stem

    age = None
    age_match = AGE_RE.search(stem)
    if age_match:
        age = age_match.group(1)
        stem_wo_age = stem[:age_match.start()] + " " + stem[age_match.end():]
    else:
        stem_wo_age = stem

    raw_tokens = re.split(r"[\s_-]+", stem_wo_age)
    raw_tokens = [t.strip() for t in raw_tokens if t.strip()]

    measure = metric = category = None
    for tok in raw_tokens:
        if canonicalize(tok, MEASURE_MAP):
            measure = canonicalize(tok, MEASURE_MAP)
            continue
        if canonicalize(tok, METRIC_MAP):
            metric = canonicalize(tok, METRIC_MAP)
            continue
        if canonicalize(tok, CATEGORY_MAP):
            category = canonicalize(tok, CATEGORY_MAP)
            continue

    if category is None:
        for part in path.parts:
            c = canonicalize(part, CATEGORY_MAP)
            if c:
                category = c
                break

    if measure is None:
        for part in path.parts:
            m = canonicalize(part, MEASURE_MAP)
            if m:
                measure = m
                break

    if not all([measure, metric, category, age]):
        return None

    return {"measure": measure, "metric": metric, "category": category,
            "age_from_filename": age}


def get_age_from_content(path: Path):
    """Read the age_name column from the first data row."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                age_name = row.get("age_name", "")
                m = AGE_RE.search(age_name)
                return m.group(1) if m else None
    except Exception as e:
        return f"ERROR: {e}"
    return None


# ---- STEP 1: normalize -------------------------------------------------

def step1_normalize(raw_root: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)

    unparsed, age_mismatches, invalid_age = [], [], []
    csv_files = sorted(raw_root.rglob("*.csv"))
    print(f"[STEP 1] Found {len(csv_files)} CSV files under {raw_root}")

    planned = []
    target_sources = defaultdict(list)

    for path in csv_files:
        parsed = parse_filename(path)
        if parsed is None:
            unparsed.append(str(path))
            continue

        if parsed["age_from_filename"] not in VALID_AGE_BANDS:
            invalid_age.append((str(path), parsed["age_from_filename"]))

        content_age = get_age_from_content(path)
        final_age = parsed["age_from_filename"]

        if content_age and content_age != parsed["age_from_filename"]:
            age_mismatches.append({"file": str(path),
                                    "filename_age": parsed["age_from_filename"],
                                    "content_age": content_age})
            final_age = content_age

        canonical_name = (f"{parsed['measure']} - {parsed['metric']} - "
                           f"{parsed['category']} - {final_age}.csv")
        target_dir = out_root / parsed["measure"] / parsed["category"]
        target_path = target_dir / canonical_name

        target_sources[str(target_path)].append(str(path))
        planned.append({"source": path, "target": target_path})

    collisions = {t: s for t, s in target_sources.items() if len(s) > 1}
    written, quarantined = [], []

    for item in planned:
        source, target = item["source"], item["target"]
        target.parent.mkdir(parents=True, exist_ok=True)

        if str(target) in collisions:
            safe_name = f"{target.stem} [CONFLICT - from {source.stem}]{target.suffix}"
            safe_path = target.parent / safe_name
            safe_path.write_bytes(source.read_bytes())
            quarantined.append(str(safe_path))
        else:
            target.write_bytes(source.read_bytes())
            written.append(str(target))

    print(f"[STEP 1] Written cleanly: {len(written)} files")

    if unparsed:
        print(f"\n[STEP 1][UNPARSED] {len(unparsed)} files could not be parsed:")
        for f in unparsed:
            print("   ", f)

    if invalid_age:
        print(f"\n[STEP 1][INVALID AGE IN FILENAME] {len(invalid_age)} files:")
        for f, age in invalid_age:
            print(f"    {f} -> parsed age '{age}'")

    if age_mismatches:
        print(f"\n[STEP 1][FILENAME vs CONTENT MISMATCH] {len(age_mismatches)} files "
              f"(content age was used for output name):")
        for m in age_mismatches:
            print(f"    {m['file']}")
            print(f"        filename said: {m['filename_age']}   content says: {m['content_age']}")

    if collisions:
        print(f"\n[STEP 1][COLLISIONS — NOTHING OVERWRITTEN] {len(collisions)} canonical "
              f"filename(s) claimed by more than one source. Quarantined with "
              f"'[CONFLICT - from ...]' suffix, resolved in step 2 below.")

    print(f"\n[STEP 1] Total on disk after step 1: {len(written) + len(quarantined)} "
          f"({len(written)} clean + {len(quarantined)} quarantined) "
          f"— should equal {len(csv_files) - len(unparsed)}.")

    return {"written": written, "quarantined": quarantined,
            "unparsed": unparsed, "invalid_age": invalid_age,
            "age_mismatches": age_mismatches, "collisions": collisions}


# ---- STEP 2: resolve conflicts -----------------------------------------

def step2_resolve(out_root: Path):
    conflict_files = sorted(out_root.rglob("*[CONFLICT*"))

    print(f"\n[STEP 2] Found {len(conflict_files)} conflict files to resolve.")
    if not conflict_files:
        print("[STEP 2] Nothing to resolve.")
        return

    groups = {}
    for path in conflict_files:
        m = CONFLICT_RE.match(path.name)
        if not m:
            print(f"[STEP 2] SKIP (unexpected name format): {path}")
            continue
        canonical_name = f"{m.group(1)}.csv"
        groups.setdefault((path.parent, canonical_name), []).append(path)

    resolved, unresolved = 0, 0

    for (parent_dir, canonical_name), files in groups.items():
        canonical_age_match = AGE_RE.search(canonical_name)
        canonical_age = canonical_age_match.group(1) if canonical_age_match else None
        canonical_path = parent_dir / canonical_name

        print(f"\n[STEP 2] --- {canonical_path} ---")
        print(f"    canonical age band expected: {canonical_age}")

        matches = []
        for f in files:
            content_age = get_age_from_content(f)
            print(f"    {f.name}")
            print(f"        content age_name says: {content_age}")
            if content_age == canonical_age:
                matches.append(f)

        if len(matches) == 1:
            winner = matches[0]
            loser = [f for f in files if f != winner][0]

            repull_dir = parent_dir / "_needs_repull"
            repull_dir.mkdir(exist_ok=True)

            shutil.copy2(winner, canonical_path)
            shutil.move(str(loser), str(repull_dir / loser.name))
            winner.unlink()

            print(f"    RESOLVED: '{winner.name}' -> written as canonical '{canonical_name}'")
            print(f"    MOVED (real gap, needs GBD re-pull): '{loser.name}' -> {repull_dir}")
            resolved += 1
        else:
            reason = "neither file matches the expected age band" if not matches \
                else "more than one file matches (unexpected)"
            print(f"    UNRESOLVED ({reason}) — left as-is, inspect manually.")
            unresolved += 1

    print(f"\n[STEP 2] Summary: {resolved} auto-resolved, {unresolved} need manual review.")


def main(raw_root, out_root):
    raw_root, out_root = Path(raw_root), Path(out_root)
    step1_normalize(raw_root, out_root)
    step2_resolve(out_root)
    print("\nDone. Check any '_needs_repull' folders and any [STEP 2] UNRESOLVED "
          "entries above before treating the dataset as complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python normalize_and_resolve.py <raw_root_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])