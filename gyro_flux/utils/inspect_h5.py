"""
Simple HDF5 file inspector.

Prints all groups, datasets, their shapes, dtypes, and attributes.
No command-line arguments - just set FILE_PATH below and run.

Usage:
    python gyro_flux/utils/inspect_h5.py
"""

import h5py
import numpy as np

# ============================================================
# SET YOUR FILE PATH HERE
# ============================================================
FILE_PATH = "/home/guzmans/fundiff/data/2019_03-isotope_d_a2_filtered/2019_03-isotope_d_a2_filtered.h5"
FOLDER_PATH = "/home/guzmans/fundiff/data/"
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

def print_h5_times():
    """Recursively scan all H5 files in FOLDER_PATH and print histogram of first time values.
    
    Expected structure:
        FOLDER_PATH/
            run1_folder/
                run1.h5
            run2_folder/
                run2.h5
            ...
    """
    from pathlib import Path
    from collections import Counter
    
    folder_path = Path(FOLDER_PATH)
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {FOLDER_PATH}")
        return
    
    # Find all H5 files recursively (searches in all subdirectories)
    # This handles: folder_path/run_folder/run.h5 structure
    h5_files = list(folder_path.rglob("*.h5"))
    
    if len(h5_files) == 0:
        print(f"No H5 files found in {FOLDER_PATH}")
        return
    
    print(f"Found {len(h5_files)} H5 files")
    print("Scanning first time values...")
    print()
    
    first_values = []
    times_lengths = []
    end_times = []  # Track last time value for each file
    errors = []
    # Map first time value to list of (file_path, times_array) for all files
    first_val_to_files = {}
    
    for h5_file in h5_files:
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'times' in f:
                    times = f['times'][:]
                    times_len = len(times)
                    if times_len > 0:
                        first_val = float(times[0])
                        last_val = float(times[-1])  # End time
                        first_values.append(first_val)
                        times_lengths.append(times_len)
                        end_times.append(last_val)
                        # Store all files for this first value
                        if first_val not in first_val_to_files:
                            first_val_to_files[first_val] = []
                        first_val_to_files[first_val].append((h5_file, times))
                    else:
                        errors.append(f"{h5_file.name}: empty times array")
                else:
                    errors.append(f"{h5_file.name}: no 'times' dataset")
        except Exception as e:
            errors.append(f"{h5_file.name}: {str(e)}")
    
    # Count occurrences
    counter = Counter(first_values)
    
    # Print histogram
    print("=" * 60)
    print("Histogram of First Time Values")
    print("=" * 60)
    
    if counter:
        # Sort by value for readability
        for first_val in sorted(counter.keys()):
            count = counter[first_val]
            print(f"{first_val}: {count} files")
        
        print()
        print(f"Total files processed: {len(first_values)}")
    else:
        print("No valid first time values found.")
    
    # Print first 3 time values for each unique starting time and check consistency
    if first_val_to_files:
        print()
        print("=" * 60)
        print("First 3 Time Values for Each Starting Time")
        print("=" * 60)
        
        for first_val in sorted(first_val_to_files.keys()):
            files_list = first_val_to_files[first_val]
            # Use first file as example
            example_file, example_times = files_list[0]
            num_to_show = min(3, len(example_times))
            example_first_three = example_times[:num_to_show]
            
            print(f"\nOption starting at {first_val}:")
            print(f"  Example file: {example_file.name}")
            print(f"  First {num_to_show} time values: ", end="")
            print(", ".join([f"{val:.6f}" for val in example_first_three]))
            
            # Check if all files with this starting time have the same first 3 values
            all_match = True
            mismatch_groups = {}  # Map unique first 3 values to list of files
            short_sequences = []  # Files with sequences too short
            
            for h5_file, times in files_list[1:]:  # Check all other files
                if len(times) < num_to_show:
                    all_match = False
                    short_sequences.append(f"{h5_file.name} (len={len(times)})")
                else:
                    file_first_three = times[:num_to_show]
                    # Compare with tolerance for floating point (allow differences up to 1e-5)
                    if not np.allclose(file_first_three, example_first_three, rtol=1e-5, atol=1e-5):
                        all_match = False
                        # Use tuple of rounded values as key for grouping
                        key = tuple(np.round(file_first_three, 6))
                        if key not in mismatch_groups:
                            mismatch_groups[key] = []
                        mismatch_groups[key].append(h5_file.name)
            
            if all_match:
                print(f"  ✓ All {len(files_list)} files with this starting time share the same first {num_to_show} values")
            else:
                print(f"  ✗ WARNING: Not all files match! ({len(files_list)} total files)")
                print(f"    Expected: {[f'{v:.6f}' for v in example_first_three]}")
                
                # Show short sequences first if any
                if short_sequences:
                    print(f"    Short sequences ({len(short_sequences)} files):")
                    for seq in short_sequences:
                        print(f"      - {seq}")
                
                # Group and show mismatches by their unique first 3 values
                if mismatch_groups:
                    print(f"    Mismatch groups ({len(mismatch_groups)} unique patterns):")
                    for key, files in sorted(mismatch_groups.items(), key=lambda x: len(x[1]), reverse=True):
                        values_str = [f"{v:.6f}" for v in key]
                        print(f"      Pattern [{', '.join(values_str)}]: {len(files)} files")
                        #for file_name in sorted(files):
                        #    print(f"        - {file_name}")
    
    # Print times array length statistics
    if times_lengths:
        min_length = min(times_lengths)
        max_length = max(times_lengths)
        print()
        print("=" * 60)
        print("Times Array Length Statistics")
        print("=" * 60)
        print(f"Lowest length: {min_length}")
        print(f"Highest length: {max_length}")
        if len(times_lengths) > 1:
            print(f"Length range: {max_length - min_length}")
    
    # Print end time statistics
    if end_times:
        min_end_time = min(end_times)
        max_end_time = max(end_times)
        print()
        print("=" * 60)
        print("End Time Statistics")
        print("=" * 60)
        print(f"Lowest end time: {min_end_time:.6f}")
        print(f"Highest end time: {max_end_time:.6f}")
        if len(end_times) > 1:
            print(f"End time range: {max_end_time - min_end_time:.6f}")
    
    # Print errors if any
    if errors:
        print()
        print(f"Errors ({len(errors)} files):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

def print_all_file_times():
    """Print all time values for each H5 file in FOLDER_PATH.
    
    For each file, prints:
    - file: file_name
    - time: first_val, second_val, ..., last_val (n_time_steps timesteps)
    
    Values are rounded to 3 decimal places.
    """
    from pathlib import Path
    
    folder_path = Path(FOLDER_PATH)
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {FOLDER_PATH}")
        return
    
    # Find all H5 files recursively
    h5_files = list(folder_path.rglob("*.h5"))
    
    if len(h5_files) == 0:
        print(f"No H5 files found in {FOLDER_PATH}")
        return
    
    print(f"Processing {len(h5_files)} H5 files...")
    print()
    
    errors = []
    
    for h5_file in sorted(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'times' in f:
                    times = f['times'][:]
                    times_len = len(times)
                    
                    if times_len > 0:
                        # Round to 3 decimals
                        times_rounded = np.round(times, 3)
                        
                        # Format: first 3 values, ..., last value
                        if times_len <= 3:
                            # Show all values if 3 or fewer
                            times_str = ", ".join([f"{val:.3f}" for val in times_rounded])
                        else:
                            # Show first 3, ..., last
                            first_three = ", ".join([f"{val:.3f}" for val in times_rounded[:3]])
                            last_val = f"{times_rounded[-1]:.3f}"
                            times_str = f"{first_three}, ..., {last_val}"
                        
                        print(f"file: {h5_file.name}")
                        print(f"time: {times_str} ({times_len} timesteps)")
                        print()
                    else:
                        errors.append(f"{h5_file.name}: empty times array")
                else:
                    errors.append(f"{h5_file.name}: no 'times' dataset")
        except Exception as e:
            errors.append(f"{h5_file.name}: {str(e)}")
    
    # Print errors if any
    if errors:
        print()
        print(f"Errors ({len(errors)} files):")
        for error in errors:
            print(f"  {error}")

def check_negative():
    """Check for negative energy flux values in the first 10 timesteps.
    
    Scans all H5 files recursively and checks if any of the first 10 energy flux
    values are negative for either ion or electron species.
    """
    from pathlib import Path
    
    folder_path = Path(FOLDER_PATH)
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {FOLDER_PATH}")
        return
    
    # Find all H5 files recursively
    h5_files = list(folder_path.rglob("*.h5"))
    
    if len(h5_files) == 0:
        print(f"No H5 files found in {FOLDER_PATH}")
        return
    
    print(f"Checking {len(h5_files)} H5 files for negative energy flux values...")
    print()
    
    files_with_negatives = 0
    files_without_negatives = 0
    errors = []
    
    for h5_file in sorted(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check if required datasets exist
                has_ions = 't0_fields/total_flux_species0' in f
                has_electrons = 't0_fields/total_flux_species1' in f
                
                if not has_ions and not has_electrons:
                    errors.append(f"{h5_file.name}: missing flux datasets")
                    continue
                
                # Load first 10 energy flux values for each species
                ions_negative = False
                electrons_negative = False
                ions_all_negative = False
                electrons_all_negative = False
                ions_some_negative = False
                electrons_some_negative = False
                num_timesteps = None
                
                if has_ions:
                    flux_species0 = f['t0_fields/total_flux_species0'][:]  # Shape: (1, n_time, 3)
                    energy_flux_ions = flux_species0[0, :, 1]  # Shape: (n_time,)
                    num_timesteps = len(energy_flux_ions)
                    first_10_ions = energy_flux_ions[:10]
                    negative_ions = first_10_ions < 0
                    ions_negative = np.any(negative_ions)
                    if ions_negative:
                        ions_all_negative = np.all(negative_ions)
                        ions_some_negative = not ions_all_negative
                
                if has_electrons:
                    flux_species1 = f['t0_fields/total_flux_species1'][:]  # Shape: (1, n_time, 3)
                    energy_flux_electrons = flux_species1[0, :, 1]  # Shape: (n_time,)
                    if num_timesteps is None:
                        num_timesteps = len(energy_flux_electrons)
                    first_10_electrons = energy_flux_electrons[:10]
                    negative_electrons = first_10_electrons < 0
                    electrons_negative = np.any(negative_electrons)
                    if electrons_negative:
                        electrons_all_negative = np.all(negative_electrons)
                        electrons_some_negative = not electrons_all_negative
                
                # Fallback to times array if num_timesteps still not set
                if num_timesteps is None and 'times' in f:
                    num_timesteps = len(f['times'][:])
                
                # Print warning if any negatives found
                if ions_negative or electrons_negative:
                    files_with_negatives += 1
                    
                    # Determine which species have negatives
                    species_list = []
                    if ions_negative:
                        if ions_all_negative:
                            species_list.append("ions (all negative)")
                        else:
                            species_list.append("ions (some negative)")
                    if electrons_negative:
                        if electrons_all_negative:
                            species_list.append("electrons (all negative)")
                        else:
                            species_list.append("electrons (some negative)")
                    
                    species_str = " and ".join(species_list)
                    timesteps_str = f" - {num_timesteps} timesteps" if num_timesteps is not None else ""
                    print(f"WARNING: {h5_file.name} has negative energy flux values in first 10 timesteps - {species_str}{timesteps_str}")
                else:
                    files_without_negatives += 1
                    
        except Exception as e:
            errors.append(f"{h5_file.name}: {str(e)}")
    
    # Print summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files with no negative values: {files_without_negatives}")
    print(f"Files with negative values: {files_with_negatives}")
    
    # Print errors if any
    if errors:
        print()
        print(f"Errors ({len(errors)} files):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

if __name__ == "__main__":
    print_h5_times()


