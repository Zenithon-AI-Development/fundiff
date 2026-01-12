"""
Simple NPZ file inspector.

Prints all arrays, their shapes, dtypes, and previews.
Handles pickled objects (dicts, etc.) with allow_pickle=True.
No command-line arguments - just set FILE_PATH below and run.

Usage:
    python gyro_flux/utils/inspect_npz.py
"""

import numpy as np

# ============================================================
# SET YOUR FILE PATH HERE
# ============================================================
FILE_PATH = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF/2019_03-isotope_h_a4_filtered/conditioning_data.npz"
# ============================================================


def preview_array(arr, max_elements=10):
    """Create a preview string for an array."""
    if arr.size == 0:
        return "[]"
    
    flat = arr.ravel()
    if flat.size <= max_elements:
        return repr(flat.tolist())
    
    head = flat[:5].tolist()
    tail = flat[-5:].tolist()
    return f"[{head[0]}, {head[1]}, ... ({arr.size} total) ..., {tail[-2]}, {tail[-1]}]"


def inspect_object(obj, indent=""):
    """Inspect a Python object (dict, list, etc.)."""
    if isinstance(obj, dict):
        print(f"{indent}  type: dict with {len(obj)} keys")
        print(f"{indent}  keys: {sorted(obj.keys())}")
        for key in sorted(obj.keys()):
            val = obj[key]
            if isinstance(val, np.ndarray):
                print(f"{indent}    {key}: ndarray, shape={val.shape}, dtype={val.dtype}")
                print(f"{indent}      preview: {preview_array(val)}")
            else:
                val_repr = repr(val)
                if len(val_repr) > 80:
                    val_repr = val_repr[:80] + "..."
                print(f"{indent}    {key}: {type(val).__name__} = {val_repr}")
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}  type: {type(obj).__name__} with {len(obj)} elements")
        for i, item in enumerate(obj[:5]):
            item_repr = repr(item)
            if len(item_repr) > 60:
                item_repr = item_repr[:60] + "..."
            print(f"{indent}    [{i}]: {item_repr}")
        if len(obj) > 5:
            print(f"{indent}    ... and {len(obj) - 5} more")
    else:
        obj_repr = repr(obj)
        if len(obj_repr) > 100:
            obj_repr = obj_repr[:100] + "..."
        print(f"{indent}  type: {type(obj).__name__}")
        print(f"{indent}  value: {obj_repr}")


def main():
    print("=" * 60)
    print("NPZ File Inspector")
    print("=" * 60)
    print(f"File: {FILE_PATH}")
    print()
    
    try:
        with np.load(FILE_PATH, allow_pickle=True) as npz:
            keys = sorted(list(npz.keys()))
            print(f"Number of arrays: {len(keys)}")
            print(f"Keys: {keys}")
            print()
            print("-" * 60)
            
            for key in keys:
                arr = npz[key]
                print(f"\n[{key}]")
                print(f"  shape: {arr.shape}")
                print(f"  dtype: {arr.dtype}")
                
                # Handle object arrays (pickled Python objects)
                if arr.dtype == object:
                    try:
                        # Try to extract the object
                        if arr.ndim == 0:
                            obj = arr.item()
                        elif arr.size == 1:
                            obj = arr.ravel()[0]
                        else:
                            obj = arr
                        inspect_object(obj, "")
                    except Exception as e:
                        print(f"  (could not inspect object: {e})")
                else:
                    # Regular numeric array
                    print(f"  min: {arr.min():.6g}" if arr.size > 0 else "  min: N/A")
                    print(f"  max: {arr.max():.6g}" if arr.size > 0 else "  max: N/A")
                    print(f"  preview: {preview_array(arr)}")
                    
    except FileNotFoundError:
        print(f"ERROR: File not found: {FILE_PATH}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

