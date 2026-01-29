"""
Compare similarity of runs within a parameter space vs across all parameter spaces.

Analyzes whether runs within the same param space (e.g., "2019_03-isotope") are more
similar to each other than runs across all param spaces, which could explain why
training on a single param space performs differently than training on all data.

Usage:
    python gyro_flux/utils/param_space_vs_all.py
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================
# CONFIGURATION
# ============================================================
PARAM_SPACE_NAMES = ["2019_03-isotope_t", "2019_03-isotope_d", "2019_03-isotope_h", "2019_05-garyshift", "2019_11-exb_2res", "2019_11-exb_kappa", "2019_11-exb_shift", "2020_08-nuei", "r90_dlntdr", "STD_A"]  # List of substrings to filter folder names
DATA_PATH = Path("/home/guzmans/fundiff/data/")
EXCLUDE_PATTERN = "2023_04-exb_paper"  # Always exclude these folders
# ============================================================


@dataclass
class RunStats:
    """Statistics for a single run."""
    folder_name: str
    n_timesteps: int
    tau_min: float
    tau_max: float
    tau_range: float
    
    # Growth phase analysis (simple heuristic: where flux is increasing)
    growth_phase_length: int  # Number of timesteps in growth phase
    growth_phase_tau: float   # Duration in tau units
    
    # Saturation phase analysis (after growth)
    saturation_mean_ion: float
    saturation_std_ion: float
    saturation_mean_electron: float
    saturation_std_electron: float
    
    # Overall statistics
    mean_flux_ion: float
    mean_flux_electron: float
    std_flux_ion: float
    std_flux_electron: float
    
    # Jaggedness (mean absolute difference between consecutive timesteps)
    jaggedness_ion: float
    jaggedness_electron: float


def load_flux_data(h5_file: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load flux and time data from H5 file.
    
    Returns:
        (times, flux_ion, flux_electron) or None if error
        - times: (n_time,) normalized tau values
        - flux_ion: (n_time,) energy flux for species 0
        - flux_electron: (n_time,) energy flux for species 1
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            # Load times
            if 'times' not in f:
                return None
            times_raw = f['times'][:]
            
            # Normalize to tau = (t - 3.0) / 1000.0
            tau = (times_raw - 3.0) / 1000.0
            
            # Load flux data
            if 't0_fields/total_flux_species0' not in f or 't0_fields/total_flux_species1' not in f:
                print(f"Missing flux datasets in {h5_file.name}")
                return None
            
            flux_species0 = f['t0_fields/total_flux_species0'][:]  # (1, n_time, 3)
            flux_species1 = f['t0_fields/total_flux_species1'][:]  # (1, n_time, 3)
            
            # Extract energy flux (index 1)
            flux_ion = flux_species0[0, :, 1]      # (n_time,)
            flux_electron = flux_species1[0, :, 1]  # (n_time,)
            
            return tau, flux_ion, flux_electron
            
    except Exception as e:
        print(f"Error loading {h5_file.name}: {e}")
        return None


def find_growth_phase_end(flux: np.ndarray) -> int:
    """Find the end of growth phase using a simple, robust heuristic.
    
    Growth phase ends when:
    1. Flux has reached >70% of saturation level (median of middle 30-70%)
    2. Two consecutive timesteps are not increasing
    
    Returns index where saturation begins.
    """
    n = len(flux)
    if n < 100:
        return n // 2  # Default to halfway if too short
    
    # Use middle portion (30-70%) as saturation reference
    saturation_reference = flux[int(0.3 * n):int(0.7 * n)]
    saturation_level = np.median(saturation_reference)
    target_level = 0.7 * saturation_level
    
    # Find first point where flux is above target AND next 2 timesteps don't increase
    for i in range(n - 1):
        if flux[i] > target_level:
            # Check if next 2 timesteps don't increase (growth has stopped)
            if flux[i+1] <= flux[i]:
                return max(i, n // 50)  # At least 2% into trajectory
    
    # Fallback: if no clear transition, use conservative estimate
    return min(n // 4, 100)  # First 25% or first 100 points


def compute_run_stats(folder_name: str, tau: np.ndarray, flux_ion: np.ndarray, 
                     flux_electron: np.ndarray) -> RunStats:
    """Compute statistics for a single run."""
    
    n_timesteps = len(tau)
    tau_min = float(tau[0])
    tau_max = float(tau[-1])
    tau_range = tau_max - tau_min
    
    # Find growth phase end (use average of both species)
    growth_end_ion = find_growth_phase_end(flux_ion)
    growth_end_electron = find_growth_phase_end(flux_electron)
    growth_end = (growth_end_ion + growth_end_electron) // 2
    
    # # DEBUG: Print details for specific runs
    # debug_runs = ["2019_03-isotope_d_a5_filtered", "2019_03-isotope_d_an6_q1_filtered"]
    # if folder_name in debug_runs:
    #     print(f"\n{'='*70}")
    #     print(f"DEBUG: {folder_name}")
    #     print(f"{'='*70}")
    #     print(f"Total timesteps: {n_timesteps}")
    #     print(f"Tau range: {tau_min:.4f} -> {tau_max:.4f}")
    #     print(f"\nGrowth phase detection:")
    #     print(f"  Ion growth end:      index {growth_end_ion:4d} (tau={tau[growth_end_ion]:.4f}, flux={flux_ion[growth_end_ion]:.2f})")
    #     print(f"  Electron growth end: index {growth_end_electron:4d} (tau={tau[growth_end_electron]:.4f}, flux={flux_electron[growth_end_electron]:.2f})")
    #     print(f"  Final growth end:    index {growth_end:4d} (tau={tau[growth_end]:.4f})")
    #     print(f"  Growth phase duration: {tau[growth_end] - tau[0]:.4f} tau units")
    #     print(f"\nSaturation reference (last 30%):")
    #     final_portion_ion = flux_ion[int(0.7 * n_timesteps):]
    #     final_portion_electron = flux_electron[int(0.7 * n_timesteps):]
    #     print(f"  Ion saturation (80th percentile):      {np.percentile(final_portion_ion, 80):.2f}")
    #     print(f"  Electron saturation (80th percentile): {np.percentile(final_portion_electron, 80):.2f}")
    #     print(f"  Ion flux at growth end:      {flux_ion[growth_end]:.2f}")
    #     print(f"  Electron flux at growth end: {flux_electron[growth_end]:.2f}")
    #     print(f"{'='*70}\n")
    
    growth_phase_length = growth_end
    growth_phase_tau = float(tau[growth_end] - tau[0])
    
    # Saturation phase statistics (after growth)
    saturation_ion = flux_ion[growth_end:]
    saturation_electron = flux_electron[growth_end:]
    
    saturation_mean_ion = float(saturation_ion.mean())
    saturation_std_ion = float(saturation_ion.std())
    saturation_mean_electron = float(saturation_electron.mean())
    saturation_std_electron = float(saturation_electron.std())
    
    # Overall statistics
    mean_flux_ion = float(flux_ion.mean())
    mean_flux_electron = float(flux_electron.mean())
    std_flux_ion = float(flux_ion.std())
    std_flux_electron = float(flux_electron.std())
    
    # Jaggedness (mean absolute consecutive difference)
    jaggedness_ion = float(np.abs(np.diff(flux_ion)).mean())
    jaggedness_electron = float(np.abs(np.diff(flux_electron)).mean())
    
    # Flag outliers: unusually long growth phases
    #if growth_phase_tau > 0.1:
    #    print(f"⚠ WARNING: {folder_name} has unusually long growth phase: {growth_phase_tau:.4f} tau units (index {growth_end}/{n_timesteps})")
    
    return RunStats(
        folder_name=folder_name,
        n_timesteps=n_timesteps,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_range=tau_range,
        growth_phase_length=growth_phase_length,
        growth_phase_tau=growth_phase_tau,
        saturation_mean_ion=saturation_mean_ion,
        saturation_std_ion=saturation_std_ion,
        saturation_mean_electron=saturation_mean_electron,
        saturation_std_electron=saturation_std_electron,
        mean_flux_ion=mean_flux_ion,
        mean_flux_electron=mean_flux_electron,
        std_flux_ion=std_flux_ion,
        std_flux_electron=std_flux_electron,
        jaggedness_ion=jaggedness_ion,
        jaggedness_electron=jaggedness_electron,
    )


def compute_inter_run_similarity(stats_list: List[RunStats]) -> Dict[str, float]:
    """Compute how similar runs are to each other.
    
    Returns dict of similarity metrics:
    - Lower values = more similar runs
    - Higher values = more diverse/heterogeneous runs
    """
    if len(stats_list) < 2:
        return {}
    
    # Extract arrays for each metric
    saturation_means_ion = np.array([s.saturation_mean_ion for s in stats_list])
    saturation_means_electron = np.array([s.saturation_mean_electron for s in stats_list])
    growth_phase_taus = np.array([s.growth_phase_tau for s in stats_list])
    jaggedness_ion = np.array([s.jaggedness_ion for s in stats_list])
    jaggedness_electron = np.array([s.jaggedness_electron for s in stats_list])
    tau_ranges = np.array([s.tau_range for s in stats_list])
    
    # Coefficient of variation (CV) = std / mean (normalized measure of dispersion)
    def cv(arr):
        mean = arr.mean()
        if mean == 0:
            return 0.0
        return float(arr.std() / np.abs(mean))
    
    similarity = {
        # How much saturation levels vary across runs (lower = more similar)
        'saturation_cv_ion': cv(saturation_means_ion),
        'saturation_cv_electron': cv(saturation_means_electron),
        
        # How much growth phase duration varies (lower = more similar)
        'growth_phase_cv': cv(growth_phase_taus),
        
        # How much jaggedness varies (lower = more similar dynamics)
        'jaggedness_cv_ion': cv(jaggedness_ion),
        'jaggedness_cv_electron': cv(jaggedness_electron),
        
        # How much time range varies (lower = more similar trajectory lengths)
        'tau_range_cv': cv(tau_ranges),
        
        # Raw standard deviations for absolute comparison
        'saturation_std_ion': float(saturation_means_ion.std()),
        'saturation_std_electron': float(saturation_means_electron.std()),
        'growth_phase_std': float(growth_phase_taus.std()),
        'tau_range_std': float(tau_ranges.std()),
    }
    
    return similarity


def analyze_dataset(data_path: Path, filter_pattern: Optional[str] = None) -> Tuple[List[RunStats], Dict[str, float]]:
    """Analyze all runs in dataset.
    
    Args:
        data_path: Path to data directory
        filter_pattern: If provided, only include folders containing this string
    
    Returns:
        (stats_list, similarity_metrics)
    """
    # Find all H5 files
    h5_files = sorted(data_path.rglob("*.h5"))
    
    # Filter by pattern and exclude unwanted folders
    if filter_pattern:
        h5_files = [f for f in h5_files if filter_pattern in f.parent.name]
    h5_files = [f for f in h5_files if EXCLUDE_PATTERN not in str(f)]
    
    print(f"Processing {len(h5_files)} H5 files...")
    
    stats_list = []
    errors = 0
    
    for h5_file in h5_files:
        data = load_flux_data(h5_file)
        if data is None:
            errors += 1
            continue
        
        tau, flux_ion, flux_electron = data
        stats = compute_run_stats(h5_file.parent.name, tau, flux_ion, flux_electron)
        stats_list.append(stats)
    
    if errors > 0:
        print(f"  {errors} files failed to load")
    
    # Compute similarity metrics
    similarity = compute_inter_run_similarity(stats_list)
    
    return stats_list, similarity


def print_summary(title: str, stats_list: List[RunStats], similarity: Dict[str, float]):
    """Print summary statistics."""
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"Number of runs: {len(stats_list)}")
    
    if len(stats_list) == 0:
        print("No runs to analyze!")
        return
    
    print()
    print("Per-Run Statistics (mean ± std):")
    print("-" * 70)
    
    # Compute means and stds across runs
    metrics = {
        'Timesteps': [s.n_timesteps for s in stats_list],
        'Tau range': [s.tau_range for s in stats_list],
        'Growth phase (tau)': [s.growth_phase_tau for s in stats_list],
        'Saturation mean (ion)': [s.saturation_mean_ion for s in stats_list],
        'Saturation mean (electron)': [s.saturation_mean_electron for s in stats_list],
        'Saturation std (ion)': [s.saturation_std_ion for s in stats_list],
        'Saturation std (electron)': [s.saturation_std_electron for s in stats_list],
        'Jaggedness (ion)': [s.jaggedness_ion for s in stats_list],
        'Jaggedness (electron)': [s.jaggedness_electron for s in stats_list],
    }
    
    for name, values in metrics.items():
        arr = np.array(values)
        print(f"{name:30s}: {arr.mean():8.3f} ± {arr.std():8.3f}  (range: [{arr.min():.3f}, {arr.max():.3f}])")
    
    print()
    print("Inter-Run Similarity Metrics (lower = more similar):")
    print("-" * 70)
    
    if similarity:
        for name, value in sorted(similarity.items()):
            # Format nicely
            if 'cv' in name:
                print(f"{name:30s}: {value:.4f}  (coefficient of variation)")
            else:
                print(f"{name:30s}: {value:.4f}  (standard deviation)")
    
    print()


def print_detailed_per_run_analysis(param_space_name: str, stats_list: List[RunStats]):
    """Print detailed per-run statistics for a parameter space.
    
    Focuses on quantities most important for training quality:
    - Growth phase characteristics (length, tau duration, flux at end)
    - Saturation phase stability (mean, std, variability)
    - Dynamics quality (jaggedness, overall flux range)
    - Training data coverage (total timesteps, tau range)
    """
    print("\n" + "=" * 70)
    print(f"DETAILED PER-RUN ANALYSIS: {param_space_name}")
    print("=" * 70)
    print(f"Total runs: {len(stats_list)}")
    print()
    
    # Print header for per-run table
    print(f"{'Run':<40} {'Timesteps':>10} {'Tau Range':>10} {'Growth τ':>10} {'Sat Mean':>10} {'Sat Std':>10} {'Jaggy':>8}")
    print("-" * 120)
    
    for stats in stats_list:
        run_name = stats.folder_name[-40:] if len(stats.folder_name) > 40 else stats.folder_name
        sat_mean = (stats.saturation_mean_ion + stats.saturation_mean_electron) / 2
        sat_std = (stats.saturation_std_ion + stats.saturation_std_electron) / 2
        jaggedness = (stats.jaggedness_ion + stats.jaggedness_electron) / 2
        
        print(f"{run_name:<40} {stats.n_timesteps:>10d} {stats.tau_range:>10.4f} {stats.growth_phase_tau:>10.4f} {sat_mean:>10.2f} {sat_std:>10.2f} {jaggedness:>8.3f}")
    
    print()
    print("Summary Statistics for Training Suitability:")
    print("-" * 70)
    
    # Critical metrics for training
    n_timesteps = np.array([s.n_timesteps for s in stats_list])
    tau_ranges = np.array([s.tau_range for s in stats_list])
    growth_phases = np.array([s.growth_phase_tau for s in stats_list])
    sat_means_ion = np.array([s.saturation_mean_ion for s in stats_list])
    sat_means_electron = np.array([s.saturation_mean_electron for s in stats_list])
    sat_stds_ion = np.array([s.saturation_std_ion for s in stats_list])
    sat_stds_electron = np.array([s.saturation_std_electron for s in stats_list])
    jaggedness_ion = np.array([s.jaggedness_ion for s in stats_list])
    jaggedness_electron = np.array([s.jaggedness_electron for s in stats_list])
    
    # Combined saturation metrics
    sat_means = (sat_means_ion + sat_means_electron) / 2
    sat_stds = (sat_stds_ion + sat_stds_electron) / 2
    jaggedness = (jaggedness_ion + jaggedness_electron) / 2
    
    print(f"Timesteps          : {n_timesteps.mean():8.1f} ± {n_timesteps.std():8.1f}  (range: [{n_timesteps.min():d}, {n_timesteps.max():d}])")
    print(f"Tau range          : {tau_ranges.mean():8.4f} ± {tau_ranges.std():8.4f}  (range: [{tau_ranges.min():.4f}, {tau_ranges.max():.4f}])")
    print(f"Growth phase (tau) : {growth_phases.mean():8.4f} ± {growth_phases.std():8.4f}  (range: [{growth_phases.min():.4f}, {growth_phases.max():.4f}])")
    print()
    print("Saturation Phase Quality:")
    print(f"  Mean flux level    : {sat_means.mean():8.2f} ± {sat_means.std():8.2f}  (range: [{sat_means.min():.2f}, {sat_means.max():.2f}])")
    print(f"  Saturation scatter : {sat_stds.mean():8.2f} ± {sat_stds.std():8.2f}  (range: [{sat_stds.min():.2f}, {sat_stds.max():.2f}])")
    print(f"  Jaggedness (noise) : {jaggedness.mean():8.3f} ± {jaggedness.std():8.3f}  (range: [{jaggedness.min():.3f}, {jaggedness.max():.3f}])")
    print()
    print("Training Suitability Notes:")
    
    # Check for problematic characteristics
    issues = []
    
    # Check growth phase uniformity
    growth_cv = growth_phases.std() / growth_phases.mean() if growth_phases.mean() > 0 else 0
    if growth_cv > 0.3:
        issues.append(f"  ⚠ Highly variable growth phases (CV={growth_cv:.3f}) - may cause training instability")
    else:
        issues.append(f"  ✓ Consistent growth phases (CV={growth_cv:.3f})")
    
    # Check saturation phase noise
    jaggedness_cv = jaggedness.std() / jaggedness.mean() if jaggedness.mean() > 0 else 0
    if jaggedness_cv > 0.5:
        issues.append(f"  ⚠ High jaggedness variability (CV={jaggedness_cv:.3f}) - noisy dynamics")
    else:
        issues.append(f"  ✓ Consistent jaggedness (CV={jaggedness_cv:.3f})")
    
    # Check saturation level variability
    sat_mean_cv = sat_stds.mean() / sat_means.mean() if sat_means.mean() > 0 else 0
    if sat_mean_cv > 0.3:
        issues.append(f"  ⚠ High saturation scatter (CV={sat_mean_cv:.3f}) - less stable targets")
    else:
        issues.append(f"  ✓ Stable saturation levels (CV={sat_mean_cv:.3f})")
    
    # Check data diversity
    tau_cv = tau_ranges.std() / tau_ranges.mean() if tau_ranges.mean() > 0 else 0
    if tau_cv < 0.1:
        issues.append(f"  ✓ Uniform trajectory lengths (CV={tau_cv:.3f})")
    else:
        issues.append(f"  ⚠ Variable trajectory lengths (CV={tau_cv:.3f})")
    
    for issue in issues:
        print(issue)
    print()


def compare_param_space_vs_all():
    """Main comparison function."""
    print("\n" + "=" * 70)
    print("PARAMETER SPACE SIMILARITY ANALYSIS")
    print("=" * 70)
    print(f"Data path: {DATA_PATH}")
    print(f"Param space filters: {PARAM_SPACE_NAMES}")
    print(f"Excluding: {EXCLUDE_PATTERN}")
    print()
    
    # Analyze each param space subset
    param_space_results = {}
    for param_space_name in PARAM_SPACE_NAMES:
        print(f"Analyzing param space: {param_space_name}")
        param_stats, param_similarity = analyze_dataset(DATA_PATH, filter_pattern=param_space_name)
        print_summary(f"PARAM SPACE: {param_space_name}", param_stats, param_similarity)
        param_space_results[param_space_name] = (param_stats, param_similarity)
    
    # Analyze all data
    print(f"\nAnalyzing all data (excluding {EXCLUDE_PATTERN})")
    all_stats, all_similarity = analyze_dataset(DATA_PATH, filter_pattern=None)
    print_summary("ALL DATA (excluding 2023_04-exb_paper)", all_stats, all_similarity)
    
    # Comparison summary for each param space and track best
    comparison_scores = {}  # Maps param_space_name -> avg ratio (lower is better/more homogeneous)
    
    for param_space_name, (param_stats, param_similarity) in param_space_results.items():
        if param_similarity and all_similarity:
            print("=" * 70)
            print(f"COMPARISON: {param_space_name} vs All Data")
            print("=" * 70)
            print("Ratio of similarity metrics (param_space / all_data)")
            print("  < 1.0 = param space is MORE similar (more homogeneous)")
            print("  > 1.0 = param space is LESS similar (more heterogeneous)")
            print("-" * 70)
            
            ratios = []
            for key in sorted(param_similarity.keys()):
                if key in all_similarity:
                    param_val = param_similarity[key]
                    all_val = all_similarity[key]
                    if all_val != 0:
                        ratio = param_val / all_val
                        comparison = "more similar" if ratio < 1.0 else "less similar"
                        print(f"{key:30s}: {ratio:.3f}  ({comparison})")
                        ratios.append(ratio)
            
            # Track average ratio for this param space (lower = more homogeneous)
            avg_ratio = np.mean(ratios) if ratios else float('inf')
            comparison_scores[param_space_name] = (avg_ratio, param_similarity)
            print()
    
    # Print best param space summary
    if comparison_scores:
        print("=" * 70)
        print("PARAMETER SPACE RANKING (by homogeneity)")
        print("=" * 70)
        print("Ranked by average ratio vs all data (lower = more homogeneous)")
        print("-" * 70)
        
        # Sort by average ratio (lower is better)
        sorted_scores = sorted(comparison_scores.items(), key=lambda x: x[1][0])
        
        for rank, (param_space_name, (avg_ratio, _)) in enumerate(sorted_scores, 1):
            # Get number of runs for this param space
            n_runs = len(param_space_results[param_space_name][0])
            print(f"{rank:2d}. {param_space_name:30s}: {avg_ratio:.4f}  ({n_runs:3d} runs)")
        
        print()
        print("=" * 70)
        print("BEST PARAMETER SPACE (Most Homogeneous)")
        print("=" * 70)
        
        best_param_space = min(comparison_scores.keys(), key=lambda x: comparison_scores[x][0])
        best_avg_ratio, best_similarity = comparison_scores[best_param_space]
        
        print(f"Winner: {best_param_space}")
        print(f"Average ratio vs all data: {best_avg_ratio:.4f} (lower = more similar)")
        print()
        print("Detailed metrics:")
        print("-" * 70)
        
        for key in sorted(best_similarity.keys()):
            if key in all_similarity:
                param_val = best_similarity[key]
                all_val = all_similarity[key]
                if all_val != 0:
                    ratio = param_val / all_val
                    print(f"{key:30s}: {param_val:.4f} vs {all_val:.4f} (ratio: {ratio:.3f})")
        print()
        
        # Print detailed per-run analysis for best param space
        best_param_stats, _ = param_space_results[best_param_space]
        print_detailed_per_run_analysis(best_param_space, best_param_stats)


if __name__ == "__main__":
    compare_param_space_vs_all()
