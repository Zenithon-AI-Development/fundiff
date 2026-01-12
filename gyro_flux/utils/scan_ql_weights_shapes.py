"""
Scan conditioning_data_description.yaml files in Well_Formatted_CGYRO_W_TGLF
and report statistics on ql_weights_reshaped field shapes.
"""

import os
from pathlib import Path
from collections import Counter
import yaml


def scan_ql_weights_shapes(base_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF"):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    
    # Get all subdirectories
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    # Check for folders missing CGYRO target data (should have <folder_name>.yaml)
    # Group by number of files in those folders
    missing_target_by_filecount = Counter()
    folders_missing_target = []
    for subdir in subdirs:
        target_yaml = subdir / f"{subdir.name}.yaml"
        if not target_yaml.exists():
            file_count = len([f for f in subdir.iterdir() if f.is_file()])
            missing_target_by_filecount[file_count] += 1
            folders_missing_target.append((subdir.name, file_count))
    
    total_scanned = 0
    files_with_field = 0
    files_without_field = 0
    shape_counter = Counter()
    files_missing_field = []
    
    for subdir in subdirs:
        yaml_file = subdir / "conditioning_data_description.yaml"
        
        if not yaml_file.exists():
            continue
        
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not parse {yaml_file}: {e}")
            continue
        
        total_scanned += 1
        
        # Check for ql_weights_reshaped field
        fields = data.get("fields", {})
        if "ql_weights_reshaped" in fields:
            files_with_field += 1
            shape = fields["ql_weights_reshaped"].get("shape", None)
            if shape is not None:
                # Convert list to tuple for hashing
                shape_tuple = tuple(shape)
                shape_counter[shape_tuple] += 1
        else:
            files_without_field += 1
            # Get last field info
            field_names = list(fields.keys())
            if field_names:
                last_field_name = field_names[-1]
                last_field_shape = fields[last_field_name].get("shape", None)
                if last_field_shape is not None:
                    shape_str = ", ".join(map(str, last_field_shape))
                else:
                    shape_str = "N/A"
            else:
                last_field_name = "N/A"
                shape_str = "N/A"
            files_missing_field.append((subdir.name, last_field_name, shape_str))
    
    # Print results
    if missing_target_by_filecount:
        for file_count, folder_count in sorted(missing_target_by_filecount.items()):
            print(f"Folders missing CGYRO target data ({file_count} files in those folders): {folder_count}")
        print()
        print("Folders missing CGYRO target data:")
        print("-" * 40)
        for name, file_count in sorted(folders_missing_target, key=lambda x: x[0]):
            print(f"  {name} ({file_count} files)")
        print()
    else:
        print("All folders have CGYRO target data.")
        print()
    print(f"Total yaml files scanned: {total_scanned}")
    print(f"Files with ql_weights_reshaped: {files_with_field}")
    print(f"Files without ql_weights_reshaped: {files_without_field}")
    print()
    print("Shape distribution:")
    print("-" * 40)
    
    # Sort by count descending
    for shape, count in sorted(shape_counter.items(), key=lambda x: -x[1]):
        shape_str = f"({', '.join(map(str, shape))})"
        print(f"shape {shape_str}: {count} files")
    
    # Print files missing the field
    if files_missing_field:
        print()
        print("Files without ql_weights_reshaped:")
        print("-" * 40)
        for name, last_field, shape_str in sorted(files_missing_field, key=lambda x: x[0]):
            print(f"  {name}, lastf: {last_field}, shape: {shape_str}")


if __name__ == "__main__":
    scan_ql_weights_shapes()

