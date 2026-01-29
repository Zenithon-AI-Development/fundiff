"""
Plot energy (heat) flux vs time for ion and electron species from CGYRO H5 files.

Usage:
    python gyro_flux/utils/plot_fluxes.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SET YOUR FOLDER PATH HERE (folder containing the H5 file)
# ============================================================
FOLDER_PATH = "/home/guzmans/fundiff/data/r70_ge_d5_g17_filtered"
# Set to True to loop through all folders starting with "2023_04-exb_paper"
LOOP_MODE = True
LOOP_FOLDER = "2019_03-isotope_d"
garyshift_growths = [30, 25, 21, 45, 31, 20, 21, 20, 18, 20, 21, 20, 21]
# ============================================================


def plot_fluxes(folder_path: str, growth_phase_time: float = None):
    """Plot energy flux vs time for both species.
    
    Args:
        folder_path: Path to folder containing the H5 file.
                    The H5 file should have the same name as the folder.
        growth_phase_time: Optional. If provided, draw a vertical line at this time
                          to mark the end of growth phase.
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"ERROR: Folder not found: {folder_path}")
        return
    
    # Get folder name and construct H5 file path
    folder_name = folder_path.name
    h5_file_path = folder_path / f"{folder_name}.h5"
    
    if not h5_file_path.exists():
        print(f"ERROR: H5 file not found: {h5_file_path}")
        return
    
    print(f"Loading data from: {h5_file_path}")
    
    # Load data from H5 file
    with h5py.File(h5_file_path, 'r') as f:
        # Load times
        if 'times' not in f:
            print("ERROR: 'times' dataset not found in H5 file")
            return
        times = f['times'][:]
        
        # Load flux data for species 0 (ions)
        if 't0_fields/total_flux_species0' not in f:
            print("ERROR: 't0_fields/total_flux_species0' dataset not found in H5 file")
            return
        flux_species0 = f['t0_fields/total_flux_species0'][:]  # Shape: (1, n_time, 3)
        # Extract energy flux (index 1) and squeeze first dimension
        energy_flux_ions = flux_species0[0, :, 1]  # Shape: (n_time,)
        
        # Load flux data for species 1 (electrons)
        if 't0_fields/total_flux_species1' not in f:
            print("ERROR: 't0_fields/total_flux_species1' dataset not found in H5 file")
            return
        flux_species1 = f['t0_fields/total_flux_species1'][:]  # Shape: (1, n_time, 3)
        # Extract energy flux (index 1) and squeeze first dimension
        energy_flux_electrons = flux_species1[0, :, 1]  # Shape: (n_time,)
    
    # Verify dimensions match
    if len(times) != len(energy_flux_ions) or len(times) != len(energy_flux_electrons):
        print(f"ERROR: Dimension mismatch - times: {len(times)}, ions: {len(energy_flux_ions)}, electrons: {len(energy_flux_electrons)}")
        return
    
    print(f"Loaded {len(times)} time points")
    
    # Create output directory (same level as gyro_flux)
    output_dir = Path(folder_path).parent.parent / "flux_plots"
    output_dir.mkdir(exist_ok=True)
    
    # Create figure with 2 subplots (ions on top, electrons on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot ions (top)
    ax1.plot(times, energy_flux_ions, 'b-', linewidth=1.5)
    if growth_phase_time is not None:
        ax1.axvline(x=growth_phase_time, color='green', linestyle='--', linewidth=2, label=f'Growth phase end (t={growth_phase_time})')
        ax1.legend(loc='upper right')
    ax1.set_ylabel('Energy Flux (Ions)', fontsize=12)
    ax1.set_title(f'Energy Flux vs Time - {folder_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot electrons (bottom)
    ax2.plot(times, energy_flux_electrons, 'r-', linewidth=1.5)
    if growth_phase_time is not None:
        ax2.axvline(x=growth_phase_time, color='green', linestyle='--', linewidth=2, label=f'Growth phase end (t={growth_phase_time})')
        ax2.legend(loc='upper right')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Energy Flux (Electrons)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / f"{folder_name}_flux.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    if LOOP_MODE:
        # Loop through all folders starting with "LOOP_FOLDER"
        data_dir = Path("/home/guzmans/fundiff/data")
        folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(LOOP_FOLDER)])
        
        if folders:
            print(f"Found {len(folders)} folders starting with {LOOP_FOLDER}")
            print()
            
            # Use garyshift_growths if looping through garyshift
            if LOOP_FOLDER == "2019_05-garyshift" and len(garyshift_growths) == len(folders):
                for i, folder in enumerate(folders):
                    print(f"Processing: {folder.name} (growth phase end: t={garyshift_growths[i]})")
                    plot_fluxes(str(folder), growth_phase_time=garyshift_growths[i])
                    print()
            else:
                for folder in folders:
                    print(f"Processing: {folder.name}")
                    plot_fluxes(str(folder))
                    print()
        else:
            print(f"No folders found starting with {LOOP_FOLDER}")
    else:
        # Plot single folder specified in FOLDER_PATH
        plot_fluxes(FOLDER_PATH)
