"""
Classify and move folders in Well_Formatted_CGYRO_W_TGLF based on (n_species, n_fields).
Folders with (2, 2) go to 2species_2fields/, others go to different_shapes/.
"""

import os
import time
import gc
from pathlib import Path
import yaml


def classify_and_move(base_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF", dry_run: bool = True):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    
    # Create target directories
    target_2s2f = base_path / "2species_2fields"
    target_different = base_path / "different_shapes"
    
    if not dry_run:
        target_2s2f.mkdir(exist_ok=True)
        target_different.mkdir(exist_ok=True)
    
    # Get all subdirectories using os.listdir to avoid Path overhead
    base_dir_str = str(base_path)
    all_items = os.listdir(base_dir_str)
    subdirs = [
        os.path.join(base_dir_str, item) for item in all_items
        if os.path.isdir(os.path.join(base_dir_str, item)) 
        and item not in ["2species_2fields", "different_shapes"]
    ]
    del all_items  # Explicitly free the list
    
    # First pass: classify all folders
    all_moves = []  # List of (subdir_path, subdir_name, destination)
    
    for idx, subdir_path in enumerate(subdirs):
        subdir_name = os.path.basename(subdir_path)
        yaml_file = os.path.join(subdir_path, "conditioning_data_description.yaml")
        
        # Default to different_shapes if can't classify
        target = target_different
        
        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                fields = data.get("fields", {})
                ql_weights = fields.get("QL_weights", {})
                metadata = ql_weights.get("metadata", {})
                
                n_species = metadata.get("n_species", None)
                n_fields = metadata.get("n_fields", None)
                
                if n_species == 2 and n_fields == 2:
                    target = target_2s2f
            except Exception as e:
                print(f"Warning: Could not parse {yaml_file}: {e}")
        
        # Store move operation with destination (use absolute paths)
        if target == target_2s2f:
            dest = os.path.abspath(os.path.join(str(target_2s2f), subdir_name))
        else:
            dest = os.path.abspath(os.path.join(str(target_different), subdir_name))
        subdir_path_abs = os.path.abspath(subdir_path)
        all_moves.append((subdir_path_abs, subdir_name, dest))
        
        if (idx + 1) % 50 == 0:
            print(f"Classified {idx + 1}/{len(subdirs)} folders...")
    
    # Count moves by type
    moved_2s2f = sum(1 for _, _, d in all_moves if str(target_2s2f) in d)
    moved_different = len(all_moves) - moved_2s2f
    
    # Create shell script with all moves (user will run it manually)
    if not dry_run:
        # Write shell script to /home/guzmans/ with full paths
        script_path = os.path.join("/home/guzmans", "move_folders.sh")
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to move folders based on classification\n")
            f.write(f"# Total moves: {len(all_moves)}\n")
            f.write(f"# To 2species_2fields: {moved_2s2f}\n")
            f.write(f"# To different_shapes: {moved_different}\n\n")
            
            for idx, (subdir_path, subdir_name, dest) in enumerate(all_moves):
                f.write(f'mv "{subdir_path}" "{dest}"\n')
                # Add delay after each move (except the last)
                if idx < len(all_moves) - 1:
                    f.write('sleep 0.1\n')
                # Progress echo every 10 moves
                if (idx + 1) % 10 == 0:
                    f.write(f'echo "Processed {idx + 1}/{len(all_moves)} moves..."\n')
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"Created shell script: {script_path}")
        print(f"Total moves: {len(all_moves)} ({moved_2s2f} to 2species_2fields, {moved_different} to different_shapes)")
        print(f"Run it manually with: bash {script_path}")
    
    print()
    print(f"{'[DRY RUN] ' if dry_run else ''}Moved to 2species_2fields: {moved_2s2f}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Moved to different_shapes: {moved_different}")
    
    # Create README
    if not dry_run:
        create_readme(base_path)
        print("Created README.md")


def create_readme(base_path: Path):
    readme_content = """# Well_Formatted_CGYRO_W_TGLF Dataset Classification

This dataset has been organized based on the (n_species, n_fields) configuration from the TGLF conditioning data.

## Folder Structure

### `2species_2fields/`
Contains **253 run folders** with the standard configuration:
- n_species = 2
- n_fields = 2

These folders are suitable for ML training with the baseline model architecture.

### `different_shapes/`
Contains **86 run folders** with non-standard configurations:
- (2, 3): 53 folders
- (3, 2): 30 folders  
- (4, 2): 3 folders

**Note:** This folder also includes **15 run folders** (those starting with `2020_04-Marinoni`) that do not contain CGYRO target data (only TGLF conditioning data is available).

## Classification Method
Classification is based on the `QL_weights.metadata.n_species` and `QL_weights.metadata.n_fields` values from each folder's `conditioning_data_description.yaml` file.
"""
    
    readme_path = base_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        print("Executing classification and moving folders...")
        classify_and_move(dry_run=False)
    else:
        print("Running in DRY RUN mode (no files will be moved)")
        print("To actually move files, run with: python classify_by_species_fields.py --execute")
        print()
        classify_and_move(dry_run=True)
