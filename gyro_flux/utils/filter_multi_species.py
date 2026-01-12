"""
Identify and optionally move multi-species TGLF conditioning folders.

Scans all conditioning_data_description.yaml files and checks how many species
are present in flux_filtered.content.keys (spec_1, spec_2, spec_3, spec_4...).

Folders with only spec_1 and spec_2 are considered "2-species" (good).
Folders with spec_3 or spec_4 are "multi-species" and can be moved.

Usage:
  # Dry run (just print which folders are multi-species):
  python gyro/filter_multi_species.py

  # Actually move multi-species folders:
  python gyro/filter_multi_species.py --move
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/home/guzmans/fundiff/well_format_filtered_cgyro_w_tglf")
YAML_BASENAME = "conditioning_data_description.yaml"
MULTI_SPECIES_DIR = "multi_species_runs"


# Regex to parse the keys: list under flux_filtered.content
RE_FLUX_FILTERED_START = re.compile(r"^\s{2}flux_filtered:\s*$")
RE_CONTENT_START = re.compile(r"^\s{4}content:\s*$")
RE_KEYS_START = re.compile(r"^\s{6}keys:\s*$")
RE_KEY_ITEM = re.compile(r"^\s{6}-\s*(\S+)\s*$")
RE_FIELD_START = re.compile(r"^\s{2}([A-Za-z0-9_.-]+):\s*$")


def parse_flux_filtered_keys(text: str) -> list[str]:
    """
    Extract the list of keys from flux_filtered.content.keys in the YAML.
    Returns an empty list if not found.
    """
    lines = text.splitlines()
    in_flux_filtered = False
    in_content = False
    in_keys = False
    keys: list[str] = []

    for line in lines:
        # Detect start of flux_filtered block
        if RE_FLUX_FILTERED_START.match(line):
            in_flux_filtered = True
            in_content = False
            in_keys = False
            continue

        # If we hit another field at same indent level, we're out of flux_filtered
        if in_flux_filtered and RE_FIELD_START.match(line) and not RE_FLUX_FILTERED_START.match(line):
            break

        if not in_flux_filtered:
            continue

        # Detect content: block
        if RE_CONTENT_START.match(line):
            in_content = True
            continue

        if not in_content:
            continue

        # Detect keys: block
        if RE_KEYS_START.match(line):
            in_keys = True
            continue

        if not in_keys:
            continue

        # Collect key items
        m = RE_KEY_ITEM.match(line)
        if m:
            keys.append(m.group(1))
            continue

        # If line doesn't match key item pattern at this indent, we're done with keys
        if line.strip() and not line.startswith("      -"):
            break

    return keys


def count_species(keys: list[str]) -> int:
    """
    Count distinct species from keys like spec_1_field_1, spec_2_field_2, etc.
    """
    species = set()
    for k in keys:
        # Extract spec_N prefix
        m = re.match(r"spec_(\d+)", k)
        if m:
            species.add(int(m.group(1)))
    return len(species)


def get_max_species(keys: list[str]) -> int:
    """
    Get the highest species number from keys.
    """
    max_spec = 0
    for k in keys:
        m = re.match(r"spec_(\d+)", k)
        if m:
            max_spec = max(max_spec, int(m.group(1)))
    return max_spec


def main(argv: list[str]) -> int:
    do_move = "--move" in argv
    root = DEFAULT_ROOT

    # Find custom root if provided (any arg that's not --move)
    for arg in argv[1:]:
        if arg != "--move":
            root = Path(arg)
            break

    print("=== filter_multi_species.py ===")
    print(f"root: {root}")
    print(f"mode: {'MOVE' if do_move else 'DRY RUN (use --move to actually move)'}")

    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        return 2

    yaml_paths = sorted(root.glob(f"*/{YAML_BASENAME}"))
    print(f"found yamls: {len(yaml_paths)}")

    if not yaml_paths:
        print("ERROR: no YAML files found.")
        return 2

    two_species: list[Path] = []
    multi_species: list[tuple[Path, int]] = []  # (folder, max_species)

    for yp in yaml_paths:
        text = yp.read_text(encoding="utf-8")
        keys = parse_flux_filtered_keys(text)
        max_spec = get_max_species(keys)

        folder = yp.parent
        if max_spec <= 2:
            two_species.append(folder)
        else:
            multi_species.append((folder, max_spec))

    print(f"\n2-species folders: {len(two_species)}")
    print(f"multi-species folders: {len(multi_species)}")

    if not multi_species:
        print("\nNo multi-species folders found. Nothing to move.")
        return 0

    # Group by species count
    by_count: dict[int, list[Path]] = {}
    for folder, n in multi_species:
        by_count.setdefault(n, []).append(folder)

    print("\nmulti-species breakdown:")
    for n in sorted(by_count.keys()):
        print(f"- {n} species: {len(by_count[n])} folders")

    print("\nmulti-species folders:")
    for folder, n in multi_species:
        print(f"- {folder.name} ({n} species)")

    if not do_move:
        print(f"\nTo move these folders to '{MULTI_SPECIES_DIR}/', run:")
        print(f"  python gyro/filter_multi_species.py --move")
        return 0

    # Actually move folders
    dest_dir = root.parent / MULTI_SPECIES_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nMoving to: {dest_dir}")

    moved = 0
    for folder, _n in multi_species:
        dest = dest_dir / folder.name
        if dest.exists():
            print(f"  SKIP (already exists): {folder.name}")
            continue
        try:
            shutil.move(str(folder), str(dest))
            print(f"  MOVED: {folder.name}")
            moved += 1
        except Exception as e:
            print(f"  ERROR moving {folder.name}: {e}")

    print(f"\nDone. Moved {moved}/{len(multi_species)} folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

