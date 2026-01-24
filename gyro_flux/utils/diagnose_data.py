"""Diagnostic script to analyze dataset heterogeneity.

Computes per-file statistics to understand what characteristics
might correlate with high reconstruction loss during training.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gyro_flux.data_utils import get_paired_folders, load_cgyro_with_times


def compute_file_statistics(folder: Path) -> Dict:
    """Compute statistics for a single CGYRO file.

    Returns dict with:
    - folder_name: str
    - n_timesteps: int
    - dt_mean: mean Δt
    - dt_std: std of Δt
    - t_start: first time value
    - t_end: last time value
    - flux0_mean, flux0_std, flux0_min, flux0_max: species 0 stats
    - flux1_mean, flux1_std, flux1_min, flux1_max: species 1 stats
    - flux0_range: max - min for species 0
    - flux1_range: max - min for species 1
    - flux0_growth_mean: mean of first 20% of species 0
    - flux0_saturated_mean: mean of last 50% of species 0
    - growth_ratio_0: ratio of saturated to growth phase
    """
    from gyro_flux.data_utils import find_h5_file

    h5_path = find_h5_file(folder)
    if h5_path is None:
        return None

    flux, times = load_cgyro_with_times(h5_path)  # (T, 2), (T,)

    n_timesteps = len(times)
    dt = np.diff(times)

    # Basic time stats
    stats = {
        'folder_name': folder.name,
        'n_timesteps': n_timesteps,
        'dt_mean': dt.mean(),
        'dt_std': dt.std(),
        't_start': times[0],
        't_end': times[-1],
        't_duration': times[-1] - times[0],
    }

    # Flux statistics for each species
    for i in range(2):
        f = flux[:, i]
        prefix = f'flux{i}'

        stats[f'{prefix}_mean'] = f.mean()
        stats[f'{prefix}_std'] = f.std()
        stats[f'{prefix}_min'] = f.min()
        stats[f'{prefix}_max'] = f.max()
        stats[f'{prefix}_range'] = f.max() - f.min()
        stats[f'{prefix}_abs_max'] = np.abs(f).max()

        # Growth vs saturation analysis
        # Assume first 20% is growth phase, last 50% is saturation
        growth_end = int(0.2 * n_timesteps)
        sat_start = int(0.5 * n_timesteps)

        growth_mean = np.abs(f[:growth_end]).mean() if growth_end > 0 else 0
        sat_mean = np.abs(f[sat_start:]).mean() if sat_start < n_timesteps else 0

        stats[f'{prefix}_growth_mean'] = growth_mean
        stats[f'{prefix}_sat_mean'] = sat_mean
        stats[f'{prefix}_growth_ratio'] = sat_mean / (growth_mean + 1e-8)

        # Jaggedness metric: mean absolute second derivative
        if n_timesteps > 2:
            d2f = np.diff(f, n=2)
            stats[f'{prefix}_jaggedness'] = np.abs(d2f).mean()
        else:
            stats[f'{prefix}_jaggedness'] = 0.0

    # Compute normalized flux variance (coefficient of variation)
    for i in range(2):
        mean = abs(stats[f'flux{i}_mean']) + 1e-8
        std = stats[f'flux{i}_std']
        stats[f'flux{i}_cv'] = std / mean  # coefficient of variation

    return stats


def analyze_dataset(data_path: str) -> Tuple[List[Dict], Dict]:
    """Analyze all files in the dataset.

    Returns:
        file_stats: List of per-file statistics
        summary: Overall summary statistics
    """
    folders = get_paired_folders(data_path)
    print(f"Found {len(folders)} paired folders")

    file_stats = []
    for folder in folders:
        stats = compute_file_statistics(folder)
        if stats is not None:
            file_stats.append(stats)

    print(f"Computed statistics for {len(file_stats)} files")

    # Compute summary statistics
    if len(file_stats) == 0:
        return [], {}

    # Convert to numpy arrays for easier analysis
    keys = [k for k in file_stats[0].keys() if k != 'folder_name']
    arrays = {k: np.array([s[k] for s in file_stats]) for k in keys}

    summary = {}
    for k, arr in arrays.items():
        summary[k] = {
            'mean': arr.mean(),
            'std': arr.std(),
            'min': arr.min(),
            'max': arr.max(),
            'median': np.median(arr),
            'p5': np.percentile(arr, 5),
            'p95': np.percentile(arr, 95),
        }

    return file_stats, summary


def find_outliers(file_stats: List[Dict], key: str, n_sigma: float = 2.0) -> List[str]:
    """Find files that are outliers for a given metric.

    Returns list of folder names that are > n_sigma standard deviations from mean.
    """
    values = np.array([s[key] for s in file_stats])
    mean = values.mean()
    std = values.std()

    outlier_mask = np.abs(values - mean) > n_sigma * std
    outlier_names = [file_stats[i]['folder_name'] for i in np.where(outlier_mask)[0]]

    return outlier_names


def print_report(file_stats: List[Dict], summary: Dict):
    """Print a diagnostic report."""

    print("\n" + "=" * 70)
    print("DATASET DIAGNOSTIC REPORT")
    print("=" * 70)

    print(f"\nTotal files: {len(file_stats)}")

    # Key metrics
    key_metrics = [
        ('n_timesteps', 'Sequence length'),
        ('dt_mean', 'Mean Δt'),
        ('t_duration', 'Time duration'),
        ('flux0_range', 'Species 0 range'),
        ('flux1_range', 'Species 1 range'),
        ('flux0_cv', 'Species 0 CV'),
        ('flux1_cv', 'Species 1 CV'),
        ('flux0_jaggedness', 'Species 0 jaggedness'),
        ('flux1_jaggedness', 'Species 1 jaggedness'),
        ('flux0_growth_ratio', 'Species 0 growth ratio'),
    ]

    print("\n" + "-" * 70)
    print("KEY METRICS SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for key, name in key_metrics:
        if key in summary:
            s = summary[key]
            print(f"{name:<25} {s['mean']:>10.3f} {s['std']:>10.3f} {s['min']:>10.3f} {s['max']:>10.3f}")

    # Find potential problematic files based on various criteria
    print("\n" + "-" * 70)
    print("OUTLIER ANALYSIS (>2σ from mean)")
    print("-" * 70)

    outlier_metrics = [
        ('flux0_range', 'High flux0 range'),
        ('flux1_range', 'High flux1 range'),
        ('flux0_jaggedness', 'High jaggedness (flux0)'),
        ('flux1_jaggedness', 'High jaggedness (flux1)'),
        ('flux0_cv', 'High coefficient of variation'),
        ('n_timesteps', 'Unusual sequence length'),
        ('dt_mean', 'Unusual Δt'),
    ]

    all_outliers = set()
    for key, name in outlier_metrics:
        outliers = find_outliers(file_stats, key, n_sigma=2.0)
        if outliers:
            print(f"\n{name} ({len(outliers)} files):")
            for f in outliers[:5]:  # Show first 5
                val = next(s[key] for s in file_stats if s['folder_name'] == f)
                print(f"  - {f}: {val:.4f}")
            if len(outliers) > 5:
                print(f"  ... and {len(outliers) - 5} more")
            all_outliers.update(outliers)

    print(f"\nTotal unique outlier files: {len(all_outliers)}")

    # Print files that appear in multiple outlier categories
    outlier_counts = {}
    for key, _ in outlier_metrics:
        for f in find_outliers(file_stats, key, n_sigma=2.0):
            outlier_counts[f] = outlier_counts.get(f, 0) + 1

    multi_outliers = [(f, c) for f, c in outlier_counts.items() if c >= 2]
    multi_outliers.sort(key=lambda x: -x[1])

    if multi_outliers:
        print("\n" + "-" * 70)
        print("FILES APPEARING IN MULTIPLE OUTLIER CATEGORIES")
        print("-" * 70)
        for f, c in multi_outliers[:10]:
            print(f"  {f}: {c} categories")

    # Correlation between metrics
    print("\n" + "-" * 70)
    print("POTENTIAL DIFFICULTY INDICATORS")
    print("-" * 70)

    # Files with extreme values
    flux0_ranges = np.array([s['flux0_range'] for s in file_stats])
    flux1_ranges = np.array([s['flux1_range'] for s in file_stats])
    jaggedness0 = np.array([s['flux0_jaggedness'] for s in file_stats])

    # Compute "difficulty score" - composite of multiple factors
    # High range + high jaggedness = hard to reconstruct
    difficulty_score = (
        (flux0_ranges / flux0_ranges.mean()) +
        (flux1_ranges / flux1_ranges.mean()) +
        (jaggedness0 / (jaggedness0.mean() + 1e-8))
    ) / 3

    print("\nTop 10 potentially difficult files (composite score):")
    sorted_indices = np.argsort(difficulty_score)[::-1]
    for i in sorted_indices[:10]:
        s = file_stats[i]
        print(f"  {s['folder_name']}: score={difficulty_score[i]:.2f}")
        print(f"    flux0_range={s['flux0_range']:.2f}, jaggedness={s['flux0_jaggedness']:.4f}")

    return all_outliers, difficulty_score


def save_statistics_csv(file_stats: List[Dict], output_path: str):
    """Save statistics to CSV for further analysis."""
    import csv

    if len(file_stats) == 0:
        return

    keys = list(file_stats[0].keys())

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(file_stats)

    print(f"\nSaved statistics to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose dataset heterogeneity")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save statistics CSV"
    )

    args = parser.parse_args()

    # Run analysis
    file_stats, summary = analyze_dataset(args.data_path)

    if len(file_stats) > 0:
        # Print report
        outliers, difficulty_scores = print_report(file_stats, summary)

        # Save to CSV if requested
        if args.output_csv:
            save_statistics_csv(file_stats, args.output_csv)
        else:
            # Default output path
            default_csv = Path(__file__).parent.parent.parent / "data" / "dataset_statistics.csv"
            default_csv.parent.mkdir(exist_ok=True)
            save_statistics_csv(file_stats, str(default_csv))
