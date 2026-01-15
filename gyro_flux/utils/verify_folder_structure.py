"""
Verify that Well_Formatted_CGYRO_W_TGLF and Well_Formatted_CGYRO_W_TGLF_structured
contain the same folders, even though the latter is organized into subdirectories.
"""

import os
from pathlib import Path
from collections import Counter


def verify_folder_structure(
    original_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF",
    structured_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured"
):
    original_path = Path(original_dir)
    structured_path = Path(structured_dir)
    
    if not original_path.exists():
        print(f"Error: {original_dir} does not exist.")
        return False
    
    if not structured_path.exists():
        print(f"Error: {structured_dir} does not exist.")
        return False
    
    # Get all folders from original (excluding classification folders if they exist)
    original_folders = set()
    for item in original_path.iterdir():
        if item.is_dir() and item.name not in ["2species_2fields", "different_shapes"]:
            original_folders.add(item.name)
    
    # Get folders from structured version
    structured_folders = set()
    
    # Check 2species_2fields
    two_species_path = structured_path / "2species_2fields"
    if two_species_path.exists():
        for item in two_species_path.iterdir():
            if item.is_dir():
                structured_folders.add(item.name)
    else:
        print(f"Warning: {two_species_path} does not exist.")
    
    # Check different_shapes
    different_shapes_path = structured_path / "different_shapes"
    if different_shapes_path.exists():
        for item in different_shapes_path.iterdir():
            if item.is_dir():
                structured_folders.add(item.name)
    else:
        print(f"Warning: {different_shapes_path} does not exist.")
    
    # Compare
    print(f"Original folder count: {len(original_folders)}")
    print(f"Structured folder count: {len(structured_folders)}")
    print()
    
    # Find differences
    only_in_original = original_folders - structured_folders
    only_in_structured = structured_folders - original_folders
    
    if only_in_original:
        print(f"Folders only in original ({len(only_in_original)}):")
        for folder in sorted(only_in_original):
            print(f"  {folder}")
        print()
    
    if only_in_structured:
        print(f"Folders only in structured ({len(only_in_structured)}):")
        for folder in sorted(only_in_structured):
            print(f"  {folder}")
        print()
    
    # Check README exists
    readme_path = structured_path / "README.md"
    if readme_path.exists():
        print("✓ README.md exists in structured directory")
    else:
        print("✗ README.md missing in structured directory")
    print()
    
    # Final verdict
    if len(original_folders) == len(structured_folders) and not only_in_original and not only_in_structured:
        print("✓ VERIFICATION PASSED: All folders match!")
        return True
    else:
        print("✗ VERIFICATION FAILED: Folders do not match")
        return False


def verify_file_counts(structured_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured"):
    structured_path = Path(structured_dir)
    
    if not structured_path.exists():
        print(f"Error: {structured_dir} does not exist.")
        return False
    
    issues = []
    checked = 0
    
    # Check both subdirectories
    for subdir_name in ["2species_2fields", "different_shapes"]:
        subdir_path = structured_path / subdir_name
        if not subdir_path.exists():
            continue
        
        for folder in subdir_path.iterdir():
            if not folder.is_dir():
                continue
            
            checked += 1
            folder_name = folder.name
            files = list(folder.iterdir())
            file_names = [f.name for f in files if f.is_file()]
            
            # Count by extension
            yaml_files = [f for f in file_names if f.endswith('.yaml')]
            h5_files = [f for f in file_names if f.endswith('.h5')]
            npz_files = [f for f in file_names if f.endswith('.npz')]
            
            is_marinoni = folder_name.startswith("2020_04-Marinoni")
            
            if is_marinoni:
                # Should have 1 yaml, 1 npz (2 files total)
                if len(file_names) != 2:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 2 files, found {len(file_names)}")
                elif len(yaml_files) != 1:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 1 yaml, found {len(yaml_files)}")
                elif len(npz_files) != 1:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 1 npz, found {len(npz_files)}")
            else:
                # Should have 2 yamls, 1 h5, 1 npz (4 files total)
                if len(file_names) != 4:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 4 files, found {len(file_names)}: {file_names}")
                elif len(yaml_files) != 2:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 2 yamls, found {len(yaml_files)}: {yaml_files}")
                elif len(h5_files) != 1:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 1 h5, found {len(h5_files)}: {h5_files}")
                elif len(npz_files) != 1:
                    issues.append(f"{subdir_name}/{folder_name}: Expected 1 npz, found {len(npz_files)}: {npz_files}")
    
    print(f"Checked {checked} folders")
    print()
    
    if issues:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ All folders have correct file counts!")
        return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--files":
        verify_file_counts()
    else:
        verify_folder_structure()
