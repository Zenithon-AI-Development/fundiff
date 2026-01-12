"""
Simple HDF5 file inspector.

Prints all groups, datasets, their shapes, dtypes, and attributes.
No command-line arguments - just set FILE_PATH below and run.

Usage:
    python gyro_flux/utils/inspect_h5.py
"""

import h5py

# ============================================================
# SET YOUR FILE PATH HERE
# ============================================================
FILE_PATH = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF/2019_03-isotope_h_a4_filtered/2019_03-isotope_h_a4_filtered.h5"
# ============================================================


def print_attrs(obj, indent=""):
    """Print attributes of an HDF5 object."""
    if len(obj.attrs) > 0:
        print(f"{indent}  attrs:")
        for key, val in obj.attrs.items():
            val_str = repr(val)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"{indent}    {key}: {val_str}")


def visit_item(name, obj, indent=""):
    """Recursively visit and print HDF5 items."""
    if isinstance(obj, h5py.Group):
        print(f"{indent}[GROUP] {name}/")
        print_attrs(obj, indent)
        for key in sorted(obj.keys()):
            child = obj[key]
            visit_item(f"{name}/{key}", child, indent + "  ")
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        chunks = obj.chunks
        compression = obj.compression
        print(f"{indent}[DATASET] {name}")
        print(f"{indent}  shape: {shape}")
        print(f"{indent}  dtype: {dtype}")
        if chunks:
            print(f"{indent}  chunks: {chunks}")
        if compression:
            print(f"{indent}  compression: {compression}")
        print_attrs(obj, indent)
        
        # Preview small datasets
        if obj.size <= 20 and obj.ndim <= 2:
            try:
                data = obj[()]
                print(f"{indent}  data: {data}")
            except Exception:
                pass


def main():
    print("=" * 60)
    print("HDF5 File Inspector")
    print("=" * 60)
    print(f"File: {FILE_PATH}")
    print()
    
    try:
        with h5py.File(FILE_PATH, "r") as f:
            print(f"File mode: {f.mode}")
            print()
            
            # Print root attributes
            print("[ROOT] /")
            print_attrs(f, "")
            print()
            
            # Visit all items
            for key in sorted(f.keys()):
                visit_item(key, f[key], "")
                print()
                
    except FileNotFoundError:
        print(f"ERROR: File not found: {FILE_PATH}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()

