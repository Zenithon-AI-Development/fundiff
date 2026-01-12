"""
Scan conditioning_data_description.yaml files in Well_Formatted_CGYRO_W_TGLF
and report histogram of (n_species, n_fields) from QL_weights metadata.
"""

from pathlib import Path
from collections import Counter
import yaml


def scan_ql_species_fields(base_dir: str = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF"):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return
    
    # Get all subdirectories, excluding folders starting with "2020_04-Marinoni_QH"
    subdirs = [
        d for d in base_path.iterdir() 
        if d.is_dir() and not d.name.startswith("2020_04-Marinoni_QH")
    ]
    
    species_fields_counter = Counter()
    total_scanned = 0
    
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
        
        # Get QL_weights metadata
        fields = data.get("fields", {})
        ql_weights = fields.get("QL_weights", {})
        metadata = ql_weights.get("metadata", {})
        
        n_species = metadata.get("n_species", None)
        n_fields = metadata.get("n_fields", None)
        
        if n_species is not None and n_fields is not None:
            species_fields_counter[(n_species, n_fields)] += 1
    
    # Print results
    print(f"Total yaml files scanned: {total_scanned}")
    print()
    print("n_species, n_fields:")
    print("-" * 40)
    
    # Sort by count descending
    for (n_species, n_fields), count in sorted(species_fields_counter.items(), key=lambda x: -x[1]):
        print(f"({n_species}, {n_fields}): {count} files")


if __name__ == "__main__":
    scan_ql_species_fields()

