import csv
import os

def load_and_print_inputs():
    """Load data from list_of_inputs.csv and print run_name and experiment columns."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'csv_stuff', 'list_of_inputs.csv')
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for row in reader:
            run_name = row[1]
            experiment = row[2]
            print(f"{run_name}, {experiment}")


def find_matching_folders():
    """For each row in CSV, find matching folder in data/ directory."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'csv_stuff', 'list_of_inputs.csv')
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    
    # Get all folders in data directory
    all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    matches_found = []
    no_matches = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header row
        
        for row in reader:
            run_name = row[1]
            experiment = row[2]
            
            # Convert experiment name: replace . with _ to match folder naming
            experiment_normalized = experiment.replace('.', '_')
            
            # First filter: find folders that start with experiment
            experiment_matches = [folder for folder in all_folders if folder.startswith(experiment_normalized)]
            
            if not experiment_matches:
                no_matches.append(f"{run_name} ({experiment}): CAN'T FIND MATCH")
                continue
            
            # Second filter: narrow down using run_name
            # Look for run_name followed by "_filtered" to avoid substring matches
            run_matches = [folder for folder in experiment_matches if f"_{run_name}_filtered" in folder]
            
            if len(run_matches) == 0:
                no_matches.append(f"{run_name} ({experiment}): CAN'T FIND MATCH")
            elif len(run_matches) == 1:
                matches_found.append(f"{run_name} ({experiment}): {run_matches[0]}")
            else:
                # Multiple matches - try to narrow down using other column values
                explanation_parts = []
                final_matches = run_matches.copy()
                
                # Check for specific patterns in folder names like q1, q3, q4
                # These might indicate variations not captured in the CSV
                special_suffixes = ['_q1_', '_q3_', '_q4_', '_hnorm_']
                
                # If some matches have special suffixes and others don't,
                # prefer the one without special suffix (the base case)
                matches_without_suffix = [f for f in run_matches if not any(suffix in f for suffix in special_suffixes)]
                matches_with_suffix = [f for f in run_matches if any(suffix in f for suffix in special_suffixes)]
                
                if matches_without_suffix and matches_with_suffix:
                    # The CSV row likely corresponds to the base case without special suffix
                    final_matches = matches_without_suffix
                    explanation_parts.append(f"choosing base case without special suffix (found variants: {', '.join([s for s in special_suffixes if any(s in f for f in matches_with_suffix)])})")
                
                if len(final_matches) == 1:
                    explanation = f" [{'; '.join(explanation_parts)}]" if explanation_parts else ""
                    matches_found.append(f"{run_name} ({experiment}): {final_matches[0]}{explanation}")
                else:
                    matches_str = ', '.join(final_matches)
                    matches_found.append(f"{run_name} ({experiment}): potential matches: {matches_str}")
    
    # Print matches first
    print(f"runs with match or potential matches: {len(matches_found)}")
    print("-----")
    for match in matches_found:
        print(match)
    
    # Then print no matches
    print(f"\nruns without match: {len(no_matches)}")
    print("-----")
    for no_match in no_matches:
        print(no_match)


def validate_and_export_parameters():
    """Validate parameters for matched runs and export to new CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'csv_stuff', 'list_of_inputs.csv')
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    output_csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'csv_stuff', 'validated_parameters.csv')
    
    # Get all folders in data directory
    all_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    # Expected values for varying parameters
    expected_DLNTDR_1 = [1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    expected_DLNNDR_1 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected_KY = [0.047, 0.055, 0.067, 0.095]
    expected_NU_EE = [0.01, 0.02, 0.05, 0.071, 0.1, 0.167, 0.278, 0.464, 0.5, 0.774, 1.0]
    expected_MASS_1 = [0.5, 1.0, 1.5, 2.0]
    
    # Helper function to find closest matching expected value
    def find_closest_match(value, expected_list, tolerance=0.01):
        """Find closest match in expected list, return canonical value or None."""
        for expected in expected_list:
            if abs(value - expected) < tolerance:
                return expected
        return None
    
    # Constant parameters (known values)
    RMAJ = 3.0
    RMIN = 0.5
    Q = 2.0
    S = 1.0
    KAPPA = 1.0
    DELTA = 0.0
    GAMMA_E = 0.0
    N_SPECIES = 2.0
    BETAE_UNIT = 0.0005
    
    validated_runs = []
    errors = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header row
        
        # Create column index mapping
        col_indices = {name: idx for idx, name in enumerate(header)}
        
        for row in reader:
            run_name = row[1]
            experiment = row[2]
            
            # Convert experiment name: replace . with _ to match folder naming
            experiment_normalized = experiment.replace('.', '_')
            
            # Find matching folder
            experiment_matches = [folder for folder in all_folders if folder.startswith(experiment_normalized)]
            if not experiment_matches:
                continue
            
            run_matches = [folder for folder in experiment_matches if f"_{run_name}_filtered" in folder]
            
            # Apply special suffix filtering
            special_suffixes = ['_q1_', '_q3_', '_q4_', '_hnorm_']
            matches_without_suffix = [f for f in run_matches if not any(suffix in f for suffix in special_suffixes)]
            
            if matches_without_suffix:
                final_matches = matches_without_suffix
            else:
                final_matches = run_matches
            
            # Only process single matches
            if len(final_matches) != 1:
                continue
            
            folder_name = final_matches[0]
            
            # Helper function to safely convert to float
            def safe_float(value, param_name):
                if value == '' or value is None:
                    raise ValueError(f"{param_name} is empty")
                return float(value)
            
            # Extract only the 5 varying parameters
            try:
                DLNTDR_1_raw = safe_float(row[col_indices['DLNTDR_1']], 'DLNTDR_1')
                DLNNDR_1_raw = safe_float(row[col_indices['DLNNDR_1']], 'DLNNDR_1')
                KY_raw = safe_float(row[col_indices['KY']], 'KY')
                NU_EE_raw = safe_float(row[col_indices['NU_EE']], 'NU_EE')
                MASS_1_raw = safe_float(row[col_indices['MASS_1']], 'MASS_1')
            except (ValueError, KeyError) as e:
                errors.append(f"Error parsing parameters for {run_name} ({experiment}) [{folder_name}]: {e}")
                continue
            
            # Map to canonical values
            DLNTDR_1 = find_closest_match(DLNTDR_1_raw, expected_DLNTDR_1)
            DLNNDR_1 = find_closest_match(DLNNDR_1_raw, expected_DLNNDR_1)
            KY = find_closest_match(KY_raw, expected_KY)
            NU_EE = find_closest_match(NU_EE_raw, expected_NU_EE)
            MASS_1 = find_closest_match(MASS_1_raw, expected_MASS_1)
            
            # Validate varying parameters
            varying_errors = []
            if DLNTDR_1 is None:
                varying_errors.append(f"DLNTDR_1={DLNTDR_1_raw} not in expected list")
            if DLNNDR_1 is None:
                varying_errors.append(f"DLNNDR_1={DLNNDR_1_raw} not in expected list")
            if KY is None:
                varying_errors.append(f"KY={KY_raw} not in expected list")
            if NU_EE is None:
                varying_errors.append(f"NU_EE={NU_EE_raw} not in expected list")
            if MASS_1 is None:
                varying_errors.append(f"MASS_1={MASS_1_raw} not in expected list")
            
            if varying_errors:
                errors.append(f"{run_name} ({experiment}) [{folder_name}]: Varying parameter errors: {', '.join(varying_errors)}")
            
            # Store validated run with constant parameters
            validated_runs.append({
                'folder_name': folder_name,
                'experiment': experiment,
                'run_name': run_name,
                'DLNTDR_1': DLNTDR_1,
                'DLNNDR_1': DLNNDR_1,
                'KY': KY,
                'NU_EE': NU_EE,
                'MASS_1': MASS_1,
                'RMAJ': RMAJ,
                'RMIN': RMIN,
                'Q': Q,
                'S': S,
                'KAPPA': KAPPA,
                'DELTA': DELTA,
                'GAMMA_E': GAMMA_E,
                'N_SPECIES': N_SPECIES,
                'BETAE_UNIT': BETAE_UNIT
            })
    
    # Print errors if any
    if errors:
        print("VALIDATION ERRORS:")
        print("=" * 80)
        for error in errors:
            print(error)
        print("=" * 80)
        print()
    
    # Print summary
    print(f"Total validated runs: {len(validated_runs)}")
    print(f"Total errors found: {len(errors)}")
    
    # Export to CSV
    if validated_runs:
        with open(output_csv_path, 'w', newline='') as f:
            fieldnames = ['folder_name', 'experiment', 'run_name', 
                         'DLNTDR_1', 'DLNNDR_1', 'KY', 'NU_EE', 'MASS_1',
                         'RMAJ', 'RMIN', 'Q', 'S', 'KAPPA', 'DELTA', 'GAMMA_E', 'N_SPECIES', 'BETAE_UNIT']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(validated_runs)
        
        print(f"\nExported validated parameters to: {output_csv_path}")
    else:
        print("\nNo validated runs to export!")


if __name__ == "__main__":
    validate_and_export_parameters()
