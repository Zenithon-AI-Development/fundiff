"""
Compare TGLF conditioning YAML descriptions across a folder and report differences.

What this checks (relative to the first YAML found):
  - Field list consistency: same number of fields and same field names
  - Per-field shape consistency: `shape:` (and `value_shape:` for scalar_object dicts)

Data layout expected:
  /home/guzmans/fundiff/well_format_filtered_cgyro_w_tglf/
    <case_1>/conditioning_data_description.yaml
    <case_2>/conditioning_data_description.yaml
    ...

Usage:
  python gyro/counting_tglf_differences.py
  python gyro/counting_tglf_differences.py /path/to/well_format_filtered_cgyro_w_tglf

Notes:
  - This repo intentionally avoids relying on PyYAML in a few small scripts,
    so we use a tiny line-based parser tailored to these YAML files.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/guzmans/fundiff/well_format_filtered_cgyro_w_tglf")
YAML_BASENAME = "conditioning_data_description.yaml"

# Keep output readable if a field mismatches in many files.
MAX_MISMATCH_FILES_PER_FIELD = 30


RE_TOPLEVEL_KEY = re.compile(r"^[A-Za-z0-9_.-]+:\s*")
RE_FIELDS_START = re.compile(r"^fields:\s*$")
RE_FIELD_START = re.compile(r"^\s{2}([A-Za-z0-9_.-]+):\s*$")
RE_SHAPE_INLINE = re.compile(r"^\s{4}shape:\s*(\S+)\s*$")
RE_SHAPE_BLOCK = re.compile(r"^\s{4}shape:\s*$")
RE_SHAPE_ITEM = re.compile(r"^\s{4}-\s*(\d+)\s*$")
RE_VALUE_SHAPE_BLOCK = re.compile(r"^\s{6}value_shape:\s*$")
RE_VALUE_SHAPE_ITEM = re.compile(r"^\s{6}-\s*(\d+)\s*$")


@dataclass(frozen=True)
class FieldShape:
    """
    Compact shape signature for comparison.
    """

    kind: str  # "dims" | "scalar" | "missing"
    dims: tuple[int, ...] = ()
    scalar: str | None = None
    value_shape: tuple[int, ...] = ()

    def to_compact_str(self) -> str:
        if self.kind == "dims":
            return f"shape={list(self.dims)}"
        if self.kind == "scalar":
            if self.scalar == "scalar_object" and self.value_shape:
                return f"shape=scalar_object value_shape={list(self.value_shape)}"
            return f"shape={self.scalar}"
        return "shape=(missing)"


def _safe_int(s: Any) -> int | None:
    try:
        return int(s)
    except Exception:
        return None


def parse_conditioning_yaml(text: str) -> tuple[list[str], dict[str, FieldShape]]:
    """
    Parse just enough of conditioning_data_description.yaml:
      fields:
        <fieldname>:
          shape: scalar_object
          # or:
          shape:
          - 21
          - 4
          # and optionally, for scalar_object dicts:
          content:
            value_shape:
            - 5
    """

    in_fields = False
    current_field: str | None = None

    # Per-field parse state
    current_shape_scalar: str | None = None
    current_shape_dims: list[int] | None = None
    current_value_shape: list[int] | None = None
    collecting_shape_dims = False
    collecting_value_shape = False

    field_order: list[str] = []
    shapes: dict[str, FieldShape] = {}

    def _finalize_current_field() -> None:
        nonlocal current_field, current_shape_scalar, current_shape_dims, current_value_shape
        nonlocal collecting_shape_dims, collecting_value_shape

        if current_field is None:
            return

        if current_shape_dims is not None:
            shapes[current_field] = FieldShape(kind="dims", dims=tuple(current_shape_dims))
        elif current_shape_scalar is not None:
            vs = tuple(current_value_shape or ())
            shapes[current_field] = FieldShape(kind="scalar", scalar=current_shape_scalar, value_shape=vs)
        else:
            shapes[current_field] = FieldShape(kind="missing")

        current_field = None
        current_shape_scalar = None
        current_shape_dims = None
        current_value_shape = None
        collecting_shape_dims = False
        collecting_value_shape = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\n")

        if not in_fields:
            if RE_FIELDS_START.match(line):
                in_fields = True
            continue

        # End of the "fields:" block when we hit the next top-level key.
        if line and not line.startswith(" ") and RE_TOPLEVEL_KEY.match(line) and not RE_FIELDS_START.match(line):
            _finalize_current_field()
            break

        m_field = RE_FIELD_START.match(line)
        if m_field:
            _finalize_current_field()
            current_field = m_field.group(1)
            field_order.append(current_field)
            continue

        if current_field is None:
            continue

        # Handle a shape: block list
        if collecting_shape_dims:
            m_item = RE_SHAPE_ITEM.match(line)
            if m_item:
                v = _safe_int(m_item.group(1))
                if v is not None and current_shape_dims is not None:
                    current_shape_dims.append(v)
                continue
            collecting_shape_dims = False
            # fall through to parse this line normally

        # Handle a value_shape: block list
        if collecting_value_shape:
            m_item = RE_VALUE_SHAPE_ITEM.match(line)
            if m_item:
                v = _safe_int(m_item.group(1))
                if v is not None and current_value_shape is not None:
                    current_value_shape.append(v)
                continue
            collecting_value_shape = False
            # fall through to parse this line normally

        m_inline = RE_SHAPE_INLINE.match(line)
        if m_inline:
            current_shape_scalar = m_inline.group(1).strip()
            continue

        if RE_SHAPE_BLOCK.match(line):
            current_shape_dims = []
            collecting_shape_dims = True
            continue

        if RE_VALUE_SHAPE_BLOCK.match(line):
            current_value_shape = []
            collecting_value_shape = True
            continue

    _finalize_current_field()
    return field_order, shapes


def _yaml_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.glob(f"*/{YAML_BASENAME}") if p.is_file()])


def report_all_field_diffs(root: Path) -> int:
    """
    Verbose report: compare all fields vs the first YAML as reference.

    Kept around because it's still useful for debugging, but `main()` now runs the
    compact `_filtered`-focused checks by default.
    """
    paths = _yaml_paths(root)

    print("=== counting_tglf_differences.py (full) ===")
    print(f"root: {root}")
    print(f"yaml_basename: {YAML_BASENAME}")
    print(f"found_yamls: {len(paths)}")

    if not paths:
        print("ERROR: no YAML files found under root.")
        return 2

    ref_path = paths[0]
    ref_text = ref_path.read_text(encoding="utf-8")
    ref_fields, ref_shapes = parse_conditioning_yaml(ref_text)
    ref_set = set(ref_fields)

    print(f"\nreference: {ref_path}")
    print(f"reference_num_fields: {len(ref_fields)}")

    # 1) Field name consistency
    field_list_ok = True
    first_field_mismatch: Path | None = None
    first_field_mismatch_details: str | None = None

    for p in paths[1:]:
        cur_text = p.read_text(encoding="utf-8")
        cur_fields, _ = parse_conditioning_yaml(cur_text)
        cur_set = set(cur_fields)

        if cur_set != ref_set or len(cur_fields) != len(ref_fields):
            field_list_ok = False
            first_field_mismatch = p
            missing = [k for k in ref_fields if k not in cur_set]
            extra = [k for k in cur_fields if k not in ref_set]
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            if not missing and not extra and cur_fields != ref_fields:
                # Same set but different order; point to the first differing index.
                for i, (a, b) in enumerate(zip(ref_fields, cur_fields)):
                    if a != b:
                        details.append(f"order_diff_at_index={i} ref={a} cur={b}")
                        break
            first_field_mismatch_details = "; ".join(details) if details else "unknown mismatch"
            break

    print("\nfield names:")
    if field_list_ok:
        print(f"- OK: all {len(ref_fields)} fields match the reference")
    else:
        print("- MISMATCH")
        print(f"  first_mismatch: {first_field_mismatch}")
        print(f"  details: {first_field_mismatch_details}")

    # 2) Shape consistency per field
    print("\nfield shapes:")
    shape_mismatches: dict[str, list[tuple[Path, FieldShape]]] = {}

    for p in paths[1:]:
        cur_text = p.read_text(encoding="utf-8")
        cur_fields, cur_shapes = parse_conditioning_yaml(cur_text)
        cur_set = set(cur_fields)

        # If fields mismatch, still try to compare intersection so we can see shape drift too.
        common = [k for k in ref_fields if k in cur_set]
        for field in common:
            ref_sig = ref_shapes.get(field, FieldShape(kind="missing"))
            cur_sig = cur_shapes.get(field, FieldShape(kind="missing"))
            if cur_sig != ref_sig:
                shape_mismatches.setdefault(field, []).append((p, cur_sig))

    if not shape_mismatches:
        print("- OK: all field shapes match the reference")
        return 0

    print(f"- MISMATCH: {len(shape_mismatches)} fields differ vs reference")
    for field in sorted(shape_mismatches.keys()):
        ref_sig = ref_shapes.get(field, FieldShape(kind="missing"))
        mism = shape_mismatches[field]
        print(f"\n- field: {field}")
        print(f"  reference: {ref_sig.to_compact_str()}")
        for i, (p, sig) in enumerate(mism[:MAX_MISMATCH_FILES_PER_FIELD]):
            print(f"  - {p.parent.name}: {sig.to_compact_str()}")
        if len(mism) > MAX_MISMATCH_FILES_PER_FIELD:
            print(f"  - ... and {len(mism) - MAX_MISMATCH_FILES_PER_FIELD} more")

    return 0


def _shape_tuple_for_compare(sig: FieldShape) -> tuple[int, ...] | None:
    """
    Convert a FieldShape into a numeric tuple when possible.

    - **dims**: returns `dims`
    - **scalar_object with value_shape**: returns `value_shape` (lets us detect drift)
    - Otherwise returns None.
    """
    if sig.kind == "dims":
        return sig.dims
    if sig.kind == "scalar" and sig.scalar == "scalar_object" and sig.value_shape:
        return sig.value_shape
    return None


def _ky_len_from_shape(sig: FieldShape) -> int | None:
    if sig.kind != "dims":
        return None
    if len(sig.dims) != 1:
        return None
    return sig.dims[0]


def report_filtered_compact(root: Path) -> int:
    """
    Compact report for dataloader-oriented invariants.

    - Only care about fields ending with `_filtered`
    - Print a histogram of ky_filtered length across all files
    - For a small set of key `_filtered` fields, determine whether shape variation
      is explained by ky only (dim0 == ky_filtered and the remaining dims are invariant),
      or whether other dims vary (e.g. eigenvalue count).
    """
    paths = _yaml_paths(root)

    print("=== counting_tglf_differences.py (filtered compact) ===")
    print(f"root: {root}")
    print(f"yaml_basename: {YAML_BASENAME}")
    print(f"found_yamls: {len(paths)}")

    if not paths:
        print("ERROR: no YAML files found under root.")
        return 2

    parsed: list[tuple[Path, dict[str, FieldShape]]] = []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        _, shapes = parse_conditioning_yaml(text)
        parsed.append((p, shapes))

    # ---- ky_filtered histogram ----
    ky_counts: dict[int | None, int] = {}
    ky_per_file: dict[Path, int | None] = {}
    for p, shapes in parsed:
        ky = _ky_len_from_shape(shapes.get("ky_filtered", FieldShape(kind="missing")))
        ky_per_file[p] = ky
        ky_counts[ky] = ky_counts.get(ky, 0) + 1

    print("\nky_filtered length histogram:")
    for ky in sorted([k for k in ky_counts.keys() if isinstance(k, int)]):
        print(f"- {ky}: {ky_counts[ky]} files")
    if None in ky_counts:
        print(f"- (missing/unknown): {ky_counts[None]} files")

    # ---- Field checks ----
    targets = [
        "flux_filtered",
        "gamma_filtered",
        "freq_filtered",
        "ql_weights_reshaped_filtered",
    ]

    print("\n_filtered field shape checks:")
    for field in targets:
        if not field.endswith("_filtered"):
            continue

        # collect signatures
        obs_counts: dict[str, int] = {}
        entries: list[tuple[Path, int | None, FieldShape, tuple[int, ...] | None]] = []
        for p, shapes in parsed:
            sig = shapes.get(field, FieldShape(kind="missing"))
            tup = _shape_tuple_for_compare(sig)
            entries.append((p, ky_per_file.get(p), sig, tup))
            obs_counts[sig.to_compact_str()] = obs_counts.get(sig.to_compact_str(), 0) + 1

        print(f"\n- {field}:")
        # Keep it compact: show up to a handful of distinct shapes.
        for s, n in sorted(obs_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
            print(f"  - {s}: {n} files")
        if len(obs_counts) > 6:
            print(f"  - ... and {len(obs_counts) - 6} more shapes")

        # Decide if field is ky-indexed by checking whether dim0 == ky_filtered for most files.
        n_match = 0
        n_check = 0
        for _p, ky, _sig, tup in entries:
            if ky is None or tup is None or len(tup) < 1:
                continue
            n_check += 1
            if tup[0] == ky:
                n_match += 1
        ky_indexed = n_check > 0 and (n_match / n_check) >= 0.8

        if not ky_indexed:
            # ky-independent: should be identical across files; if not, that's "something else".
            if len(obs_counts) == 1:
                print("  verdict: OK (ky-independent; identical across files)")
                continue
            print("  verdict: MISMATCH (ky-independent field varies across files)")
            # Show a few examples.
            shown = 0
            for p, ky, sig, _tup in entries:
                if sig.to_compact_str() != max(obs_counts, key=lambda k: obs_counts[k]):
                    print(f"  - {p.parent.name}: ky={ky} {sig.to_compact_str()}")
                    shown += 1
                    if shown >= 10:
                        break
            continue

        # ky-indexed: allow dim0 to vary with ky; tail dims must be invariant.
        tail_counts: dict[tuple[int, ...] | None, int] = {}
        bad_examples: list[str] = []
        for p, ky, sig, tup in entries:
            if tup is None or len(tup) < 1:
                tail_counts[None] = tail_counts.get(None, 0) + 1
                bad_examples.append(f"{p.parent.name}: ky={ky} {sig.to_compact_str()}")
                continue
            tail = tup[1:]
            tail_counts[tail] = tail_counts.get(tail, 0) + 1
            if ky is not None and tup[0] != ky:
                bad_examples.append(f"{p.parent.name}: ky={ky} {sig.to_compact_str()} (expected dim0==ky)")

        non_none_tails = [t for t in tail_counts.keys() if t is not None]
        unique_tails = sorted(set(non_none_tails))

        if len(unique_tails) == 1 and not bad_examples:
            print(f"  verdict: OK (ky-only variation; invariant tail={list(unique_tails[0])})")
            continue

        print("  verdict: MISMATCH (not explained by ky only)")
        if unique_tails:
            print("  tail_dims observed (counts):")
            for t in unique_tails:
                print(f"  - {list(t)}: {tail_counts.get(t, 0)} files")
        if None in tail_counts:
            print(f"  - (unparsed/missing): {tail_counts[None]} files")
        if bad_examples:
            print("  examples:")
            for ex in bad_examples[:10]:
                print(f"  - {ex}")

    return 0


def print_non80_ql_weights_reshaped_filtered(root: Path, *, expected_last_dim: int = 80) -> None:
    """
    Small helper: print which files have ql_weights_reshaped_filtered with a non-80 last dim.
    """
    field = "ql_weights_reshaped_filtered"
    paths = _yaml_paths(root)
    offenders: list[tuple[str, FieldShape]] = []
    missing = 0
    weird = 0

    for p in paths:
        text = p.read_text(encoding="utf-8")
        _, shapes = parse_conditioning_yaml(text)
        sig = shapes.get(field, FieldShape(kind="missing"))
        if sig.kind != "dims":
            if sig.kind == "missing":
                missing += 1
            else:
                weird += 1
            continue
        if len(sig.dims) != 2:
            weird += 1
            offenders.append((p.parent.name, sig))
            continue
        last = sig.dims[-1]
        if last != expected_last_dim:
            offenders.append((p.parent.name, sig))

    print(f"\n{field}: non-{expected_last_dim} last-dim offenders:")
    if not offenders:
        print("- none")
        return

    # Group by last dim for quick scanning.
    grouped: dict[int | str, list[tuple[str, FieldShape]]] = {}
    for case, sig in offenders:
        key: int | str
        if sig.kind == "dims" and len(sig.dims) >= 1:
            key = sig.dims[-1]
        else:
            key = "unknown"
        grouped.setdefault(key, []).append((case, sig))

    for key in sorted(grouped.keys(), key=lambda x: (isinstance(x, str), x)):  # ints first
        items = grouped[key]
        print(f"- last_dim={key}: {len(items)} files")
        for case, sig in sorted(items, key=lambda t: t[0]):
            print(f"  - {case}: {sig.to_compact_str()}")

    if missing:
        print(f"- missing_field: {missing} files")
    if weird:
        print(f"- non_2d_or_non_numeric_shape: {weird} files")


def check_ky_len(root: Path) -> int:
    """
    Very small check: look only at `ky` (not ky_filtered).

    Prints cases where ky length differs from the first YAML's ky length.
    Returns non-zero if any mismatches are found.
    """
    paths = _yaml_paths(root)
    print("=== counting_tglf_differences.py (ky-only) ===")
    print(f"root: {root}")
    print(f"found_yamls: {len(paths)}")
    if not paths:
        print("ERROR: no YAML files found under root.")
        return 2

    ref_path = paths[0]
    ref_text = ref_path.read_text(encoding="utf-8")
    _, ref_shapes = parse_conditioning_yaml(ref_text)
    ref_ky = _ky_len_from_shape(ref_shapes.get("ky", FieldShape(kind="missing")))

    print(f"\nreference: {ref_path.parent.name} (ky_len={ref_ky})")

    hist: dict[int | None, int] = {}
    mismatches: list[tuple[str, int | None]] = []

    for p in paths:
        text = p.read_text(encoding="utf-8")
        _, shapes = parse_conditioning_yaml(text)
        ky = _ky_len_from_shape(shapes.get("ky", FieldShape(kind="missing")))
        hist[ky] = hist.get(ky, 0) + 1
        if ky != ref_ky:
            mismatches.append((p.parent.name, ky))

    print("\nky length histogram:")
    for ky in sorted([k for k in hist.keys() if isinstance(k, int)]):
        print(f"- {ky}: {hist[ky]} files")
    if None in hist:
        print(f"- (missing/unknown): {hist[None]} files")

    if not mismatches:
        print("\nky check: OK (all files match reference ky length)")
        return 0

    print(f"\nky check: MISMATCH ({len(mismatches)} files differ from ky_len={ref_ky})")
    for case, ky in mismatches[:50]:
        print(f"- {case}: ky_len={ky}")
    if len(mismatches) > 50:
        print(f"- ... and {len(mismatches) - 50} more")

    return 1


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT_ROOT
    # Ky-only check (requested default)
    # rc = check_ky_len(root)

    # Uncomment these if you want the more detailed checks again:
    rc = report_filtered_compact(root)
    print_non80_ql_weights_reshaped_filtered(root, expected_last_dim=80)
    # rc = report_all_field_diffs(root)

    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


