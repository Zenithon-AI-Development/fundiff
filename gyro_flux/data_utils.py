"""Data loading utilities for Gyro-Flux.

This module provides dataset classes and data loaders for:
- TGLF conditioning data (conditioning_data.npz)
- CGYRO flux history (*.h5 files)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# TGLF Data Loading
# =============================================================================

def load_tglf_sample(npz_path: Path) -> np.ndarray:
    """Load and process a single TGLF conditioning file to (21, 108).
    
    Transformations:
        - QL_weights: (2, 2, 4, 21, 5) → (21, 80)
        - gamma: (21, 4) → unchanged
        - freq: (21, 4) → unchanged
        - flux_spectrum: dict of 4 × (21, 5) → (21, 20)
        - Concatenate: (21, 80+4+4+20) = (21, 108)
    
    Args:
        npz_path: Path to conditioning_data.npz
    
    Returns:
        tglf: (21, 108) array
    """
    with np.load(npz_path, allow_pickle=True) as npz:
        # QL_weights: (species=2, fields=2, eigs=4, ky=21, components=5) → (21, 80)
        ql = npz['QL_weights']  # (2, 2, 4, 21, 5)
        ql = np.moveaxis(ql, 3, 0)  # Move ky to front: (21, 2, 2, 4, 5)
        ql = ql.reshape(21, -1)  # Flatten: (21, 80)
        
        # gamma, freq: already (21, 4)
        gamma = npz['gamma']  # (21, 4)
        freq = npz['freq']    # (21, 4)
        
        # flux_spectrum: dict with 4 keys, each (21, 5) → (21, 20)
        flux_dict = npz['flux_spectrum'].item()
        flux_keys = sorted(flux_dict.keys())  # Consistent ordering
        flux_arrays = [flux_dict[k] for k in flux_keys]  # 4 × (21, 5)
        flux = np.concatenate(flux_arrays, axis=1)  # (21, 20)
        
        # Concatenate all: (21, 80+4+4+20) = (21, 108)
        tglf = np.concatenate([ql, gamma, freq, flux], axis=1)
    
    return tglf.astype(np.float32)


def get_paired_folders(base_path: str) -> List[Path]:
    """Get folders that have both TGLF (conditioning_data.npz) and CGYRO (.h5).
    
    For the new structured layout, base_path should point to:
    /home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields
    
    All folders in 2species_2fields are guaranteed to have both TGLF and CGYRO data.
    
    Args:
        base_path: Path to the data directory containing run folders
    
    Returns:
        List of folder paths with paired data
    """
    base = Path(base_path)
    paired = []
    
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        # Skip the 2023_04-exb_paper runs for now (known outliers / diagnostics)
        # This reduces the dataset from 253 to 227 folders.
        if "2023_04-exb_paper" in folder.name:
            continue
        
        # In the new structure, all folders should have both files
        # But we still verify to be safe
        has_tglf = (folder / "conditioning_data.npz").exists()
        has_cgyro = any(folder.glob("*.h5"))
        
        if has_tglf and has_cgyro:
            paired.append(folder)
        elif not has_tglf or not has_cgyro:
            # Warn if we find a folder without both (shouldn't happen in 2species_2fields)
            print(f"Warning: {folder.name} missing {'TGLF' if not has_tglf else 'CGYRO'} data")
    
    return paired


def get_tglf_only_folders(base_path: str) -> List[Path]:
    """Get all folders that have TGLF conditioning data.
    
    For the new structured layout, base_path should point to:
    /home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields
    
    Args:
        base_path: Path to the data directory
    
    Returns:
        List of folder paths with TGLF data
    """
    base = Path(base_path)
    folders = []
    
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        
        if (folder / "conditioning_data.npz").exists():
            folders.append(folder)
    
    return folders


class TGLFDataset(Dataset):
    """Dataset for TGLF conditioning data.

    Each sample is a (21, 108) array of TGLF features.

    For the new structured layout, data_path should point to:
    /home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields

    All folders in 2species_2fields are guaranteed to have both TGLF and CGYRO data.

    Args:
        data_path: Path to directory containing run folders (should be 2species_2fields)
        paired_only: If True, only include folders with both TGLF and CGYRO (default True)
        normalize: If True, apply z-score normalization per feature
        stats: Optional (mean, std) tuple for normalization. If None, computed from data.
        folder_list: Optional list of specific folders to use. If None, uses all paired/tglf folders.
    """

    def __init__(
        self,
        data_path: str,
        paired_only: bool = True,
        normalize: bool = False,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        folder_list: Optional[List[Path]] = None,
    ):
        self.data_path = Path(data_path)
        self.paired_only = paired_only
        self.normalize = normalize

        # Use provided folder list or get all folders
        if folder_list is not None:
            self.folders = folder_list
        elif paired_only:
            self.folders = get_paired_folders(data_path)
        else:
            self.folders = get_tglf_only_folders(data_path)
        
        if len(self.folders) == 0:
            raise ValueError(f"No valid folders found in {data_path}")
        
        print(f"TGLFDataset: Found {len(self.folders)} samples (paired_only={paired_only})")
        
        # Load all data into memory (small enough)
        self.data = []
        for folder in self.folders:
            npz_path = folder / "conditioning_data.npz"
            sample = load_tglf_sample(npz_path)
            self.data.append(sample)
        
        self.data = np.stack(self.data, axis=0)  # (N, 21, 108)
        
        # Normalization
        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                # Compute stats across all samples and ky modes
                self.mean = self.data.mean(axis=(0, 1), keepdims=True)  # (1, 1, 108)
                self.std = self.data.std(axis=(0, 1), keepdims=True) + 1e-8  # (1, 1, 108)
            
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
    
    def __len__(self) -> int:
        return len(self.folders)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]  # (21, 108)
    
    def get_stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return normalization stats (mean, std) if computed."""
        if self.mean is not None and self.std is not None:
            return (self.mean, self.std)
        return None
    
    def get_folder_name(self, idx: int) -> str:
        """Get the folder name for a given index."""
        return self.folders[idx].name


def create_tglf_dataloader(
    dataset: TGLFDataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for TGLF data.
    
    Args:
        dataset: TGLFDataset instance
        batch_size: Batch size (per device, will be multiplied by device count)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader instance
    """
    import jax
    num_devices = jax.device_count()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size * num_devices,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader


# =============================================================================
# CGYRO Data Loading
# =============================================================================

def find_h5_file(folder: Path) -> Optional[Path]:
    """Find the H5 file in a folder (named folder_name.h5).
    
    Args:
        folder: Path to the run folder
    
    Returns:
        Path to the H5 file, or None if not found
    """
    h5_files = list(folder.glob("*.h5"))
    if len(h5_files) == 0:
        return None
    if len(h5_files) == 1:
        return h5_files[0]
    
    # If multiple H5 files, prefer the one matching folder name
    folder_name = folder.name
    for h5_file in h5_files:
        if h5_file.stem == folder_name:
            return h5_file
    
    # Otherwise return first one
    return h5_files[0]


def load_cgyro_sample(h5_path: Path) -> np.ndarray:
    """Load CGYRO energy flux history from H5 file to (n_timesteps, 2).
    
    Extracts only the energy flux (index 1) from both species:
        - fluxes/total_flux_species0: (1, n_time, 3) → [:, :, 1] → (n_time,)
        - fluxes/total_flux_species1: (1, n_time, 3) → [:, :, 1] → (n_time,)
        - Stack to get (n_time, 2)
    
    Args:
        h5_path: Path to the H5 file
    
    Returns:
        flux: (n_timesteps, 2) array where [:, 0] is species0 and [:, 1] is species1
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        # Energy flux is at index 1 (0: particle, 1: energy, 2: momentum)
        # Shape: (1, n_time, 3) → we want (n_time,)
        flux_species0 = f['fluxes/total_flux_species0'][0, :, 1]  # (n_time,)
        flux_species1 = f['fluxes/total_flux_species1'][0, :, 1]  # (n_time,)
        
        # Stack to (n_time, 2)
        flux = np.stack([flux_species0, flux_species1], axis=1)
    
    return flux.astype(np.float32)


def load_cgyro_with_times(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load CGYRO energy flux and time array from H5 file.
    
    Args:
        h5_path: Path to the H5 file
    
    Returns:
        flux: (n_timesteps, 2) array
        times: (n_timesteps,) array of time values
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        # Energy flux is at index 1 (0: particle, 1: energy, 2: momentum)
        flux_species0 = f['fluxes/total_flux_species0'][0, :, 1]
        flux_species1 = f['fluxes/total_flux_species1'][0, :, 1]
        flux = np.stack([flux_species0, flux_species1], axis=1)

        # Robustly extract physical time coordinates, if present
        if 'times' in f:
            times = f['times'][:]
        else:
            raise KeyError(f"'times' dataset not found in H5 file: {h5_path}")
    
    return flux.astype(np.float32), times.astype(np.float32)


class CGYRODataset(Dataset):
    """Dataset for CGYRO flux time-series.

    Each sample is a (n_timesteps, 2) array of energy flux.
    Note: n_timesteps varies per sample (typically 150-3000).

    For the new structured layout, data_path should point to:
    /home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields

    All folders in 2species_2fields are guaranteed to have both TGLF and CGYRO data.

    Args:
        data_path: Path to directory containing run folders (should be 2species_2fields)
        normalize: If True, apply global normalization (z-score across all samples)
        stats: Optional (mean, std) for global normalization
        normalize_per_sample: If True, normalize each sample independently to zero mean and unit std.
                             This addresses the 324x magnitude variation across files.
                             Returns per-sample stats for denormalization at inference.
        use_padding: If True, pad all sequences to max_seq_length for efficient JAX compilation
        max_seq_length: Maximum sequence length for padding (ignored if use_padding=False)
        slice_some_time: If True, slice first slice_length timesteps from each sequence (DEPRECATED)
        slice_length: Number of timesteps to slice (only used if slice_some_time=True)
        slice_by_time: If True, slice by physical time interval instead of index count.
                       This ensures consistent physics coverage across samples with different dt.
        max_tau: Maximum normalized time τ to include (only used if slice_by_time=True).
                 τ = (t - 3.0) / 1000.0, so max_tau=0.1 means t ∈ [3.0, 103.0].
        folder_list: Optional list of specific folders to use. If None, uses all paired folders.
    """

    def __init__(
        self,
        data_path: str,
        normalize: bool = False,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        normalize_per_sample: bool = False,
        use_padding: bool = False,
        max_seq_length: int = 3000,
        slice_some_time: bool = False,
        slice_length: int = 50,
        slice_by_time: bool = False,
        max_tau: float = 0.1,
        folder_list: Optional[List[Path]] = None,
    ):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.normalize_per_sample = normalize_per_sample
        self.use_padding = use_padding
        self.max_seq_length = max_seq_length
        self.slice_some_time = slice_some_time
        self.slice_length = slice_length
        self.slice_by_time = slice_by_time
        self.max_tau = max_tau

        # Fixed time-scale and cutoff for normalized coordinates τ = (t - t0) / TIME_SCALE
        # This keeps inputs in a numerically friendly range for Fourier features.
        self.time_scale = 1000.0
        self.cutoff = 3.0
        
        # Use provided folder list or get all paired folders
        if folder_list is not None:
            self.folders = folder_list
        else:
            # Only use paired folders (those with both TGLF and CGYRO)
            self.folders = get_paired_folders(data_path)
        
        if len(self.folders) == 0:
            raise ValueError(f"No valid folders found in {data_path}")
        
        print(f"CGYRODataset: Found {len(self.folders)} samples")
        
        # Load all data into memory (variable length sequences)
        # We now keep both flux and normalized time coordinates.
        self.data = []   # list of (T_i, 2) flux arrays
        self.times = []  # list of (T_i, 1) normalized time arrays (τ)
        self.lengths = []
        
        for folder in self.folders:
            h5_path = find_h5_file(folder)
            if h5_path is None:
                raise ValueError(f"No H5 file found in {folder}")
            
            flux, times = load_cgyro_with_times(h5_path)  # flux: (T, 2), times: (T,)

            # Establish relative time so all runs start with t=3.0 as t0
            times_rel = times - self.cutoff
            # Physics-informed scaling: τ = t_rel / time_scale
            tau = times_rel / self.time_scale  # (T,)

            self.data.append(flux.astype(np.float32))
            # Store τ as (T, 1) for easier broadcasting with embeddings later
            self.times.append(tau[:, None].astype(np.float32))
            self.lengths.append(flux.shape[0])
        
        self.lengths = np.array(self.lengths)
        print(f"CGYRODataset: Timestep range [{self.lengths.min()}, {self.lengths.max()}]")
        
        # Slice strategy: take first slice_length timesteps from each sequence (DEPRECATED - use slice_by_time)
        if slice_some_time and not slice_by_time:
            # Check minimum sequence length and adjust slice_length if needed
            min_length = self.lengths.min()
            if min_length < slice_length:
                print(f"CGYRODataset: Requested slice_length={slice_length} but minimum sequence length is {min_length}")
                print(f"CGYRODataset: Adjusting slice_length to {min_length}")
                slice_length = min_length
                # Update the instance variable so it's available elsewhere
                self.slice_length = slice_length

            print(f"CGYRODataset: Slicing first {slice_length} timesteps from each sequence")

            # Slice each sample to first slice_length timesteps
            for i in range(len(self.data)):
                self.data[i] = self.data[i][:slice_length]   # (slice_length, 2)
                self.times[i] = self.times[i][:slice_length] # (slice_length, 1)

            # All sequences now have same length
            self.lengths = np.full(len(self.data), slice_length, dtype=self.lengths.dtype)
            print(f"CGYRODataset: All sequences now have fixed length {slice_length}")

        # Time-based slicing: filter by normalized time τ instead of array index
        # This ensures consistent physics coverage across samples with different time resolutions
        if slice_by_time:
            print(f"CGYRODataset: Slicing by time with max_tau={max_tau} (t ∈ [3.0, {3.0 + max_tau * 1000.0}])")

            new_data = []
            new_times = []
            new_lengths = []

            for i in range(len(self.data)):
                # τ values are stored in self.times[i] with shape (T, 1)
                tau_values = self.times[i][:, 0]  # (T,)

                # Find indices where τ <= max_tau
                valid_mask = tau_values <= max_tau
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    # Edge case: no points in range - should be rare
                    print(f"CGYRODataset: Warning - sample {i} has no points with τ <= {max_tau}, using first point")
                    valid_indices = np.array([0])

                # Slice to valid time range
                new_data.append(self.data[i][valid_indices])
                new_times.append(self.times[i][valid_indices])
                new_lengths.append(len(valid_indices))

            self.data = new_data
            self.times = new_times
            self.lengths = np.array(new_lengths)

            # Update slice_length to the minimum length after time-based slicing
            # This is used by the training script to set input dimensions
            self.slice_length = int(self.lengths.min())

            print(f"CGYRODataset: After time-based slicing:")
            print(f"  - Length range: [{self.lengths.min()}, {self.lengths.max()}]")
            print(f"  - Mean length: {self.lengths.mean():.1f}")
            print(f"  - Min length (for padding): {self.slice_length}")
        
        # Per-sample normalization: each sample normalized to zero mean, unit std
        # This addresses the 324x magnitude variation across files
        self.sample_means = None
        self.sample_stds = None

        if normalize_per_sample:
            print("CGYRODataset: Applying per-sample normalization")
            self.sample_means = []  # List of (1, 2) arrays
            self.sample_stds = []   # List of (1, 2) arrays

            for i in range(len(self.data)):
                # Compute per-sample stats (across time, per channel)
                sample_mean = self.data[i].mean(axis=0, keepdims=True)  # (1, 2)
                sample_std = self.data[i].std(axis=0, keepdims=True) + 1e-8  # (1, 2)

                self.sample_means.append(sample_mean.astype(np.float32))
                self.sample_stds.append(sample_std.astype(np.float32))

                # Normalize this sample
                self.data[i] = (self.data[i] - sample_mean) / sample_std

            # Convert to arrays for easier indexing
            self.sample_means = np.array(self.sample_means)  # (N, 1, 2)
            self.sample_stds = np.array(self.sample_stds)    # (N, 1, 2)

            # Global stats not used in per-sample mode
            self.mean = None
            self.std = None

        # Global normalization (per-channel across all samples)
        elif normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                # Compute stats across all samples and timesteps
                all_data = np.concatenate(self.data, axis=0)  # (total_timesteps, 2)
                self.mean = all_data.mean(axis=0, keepdims=True)  # (1, 2)
                self.std = all_data.std(axis=0, keepdims=True) + 1e-8  # (1, 2)

            # Apply normalization to flux; time coordinates remain as τ
            for i in range(len(self.data)):
                self.data[i] = (self.data[i] - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
        
        # If using padding, pad all sequences now
        # Determine if all sequences are fixed-length (from index-based slicing)
        all_same_length = slice_some_time and not slice_by_time

        if use_padding and not all_same_length:
            # Variable length sequences (including time-based slicing) - pad to max length
            if slice_by_time:
                # For time-based slicing, pad to max length in dataset (not max_seq_length)
                pad_length = int(self.lengths.max())
                print(f"CGYRODataset: Padding time-sliced sequences to {pad_length}")
            else:
                pad_length = max_seq_length
                print(f"CGYRODataset: Padding all sequences to {pad_length}")

            self.padded_data = np.zeros((len(self.data), pad_length, 2), dtype=np.float32)
            self.padded_times = np.zeros((len(self.data), pad_length, 1), dtype=np.float32)
            self.masks = np.zeros((len(self.data), pad_length), dtype=np.float32)

            for i, (sample, tau) in enumerate(zip(self.data, self.times)):
                seq_len = min(sample.shape[0], pad_length)
                self.padded_data[i, :seq_len, :] = sample[:seq_len]
                self.padded_times[i, :seq_len, :] = tau[:seq_len]
                self.masks[i, :seq_len] = 1.0
        elif use_padding and all_same_length:
            # When slicing by index, sequences are already fixed-length
            print(f"CGYRODataset: Sliced sequences are already fixed-length ({slice_length}), using padded_data format")
            self.padded_data = np.stack(self.data, axis=0)   # (N, slice_length, 2)
            self.padded_times = np.stack(self.times, axis=0) # (N, slice_length, 1)
            self.masks = np.ones((len(self.data), slice_length), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.folders)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_padding:
            result = {
                'cgyro': self.padded_data[idx],   # (max_seq_length or slice_length, 2)
                'time': self.padded_times[idx],   # (max_seq_length or slice_length, 1)
                'mask': self.masks[idx],          # (max_seq_length or slice_length,)
                'length': self.lengths[idx],      # original length or slice_length
            }
        else:
            result = {
                'cgyro': self.data[idx],          # (n_timesteps, 2) - variable! or (slice_length, 2) if sliced
                'time': self.times[idx],          # (n_timesteps, 1)
                'length': self.lengths[idx],
            }

        # Add per-sample normalization stats if available (for denormalization at inference)
        if self.sample_means is not None:
            result['sample_mean'] = self.sample_means[idx]  # (1, 2)
            result['sample_std'] = self.sample_stds[idx]    # (1, 2)

        return result
    
    def get_stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return normalization stats (mean, std) if computed."""
        if self.mean is not None and self.std is not None:
            return (self.mean, self.std)
        return None
    
    def get_folder_name(self, idx: int) -> str:
        """Get the folder name for a given index."""
        return self.folders[idx].name
    
    def get_length(self, idx: int) -> int:
        """Get the number of timesteps for a given index."""
        return self.lengths[idx]


def cgyro_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for CGYRO sequences.

    Handles both padded mode (use_padding=True) and variable-length mode.
    Also handles per-sample normalization stats if present.

    Args:
        batch: List of dicts from CGYRODataset.__getitem__

    Returns:
        Dictionary with 'cgyro', 'lengths', and optionally 'mask', 'sample_mean', 'sample_std' keys
    """
    # Check if we're in padded mode
    has_mask = 'mask' in batch[0]
    has_sample_stats = 'sample_mean' in batch[0]

    if has_mask:
        # Padded mode: all sequences are same length
        cgyro = np.stack([item['cgyro'] for item in batch], axis=0)
        times = np.stack([item['time'] for item in batch], axis=0)
        masks = np.stack([item['mask'] for item in batch], axis=0)
        lengths = np.array([item['length'] for item in batch])
        result = {
            'cgyro': cgyro,      # (B, max_seq_length, 2)
            'time': times,       # (B, max_seq_length, 1)
            'mask': masks,       # (B, max_seq_length)
            'lengths': lengths,  # (B,)
        }
    else:
        # Variable-length mode
        if len(batch) == 1:
            result = {
                'cgyro': batch[0]['cgyro'][None, ...],  # (1, n_timesteps, 2)
                'time': batch[0]['time'][None, ...],    # (1, n_timesteps, 1)
                'lengths': np.array([batch[0]['length']]),
            }
        else:
            # For batch_size > 1, pad to max length in batch
            lengths = [item['length'] for item in batch]
            max_len = max(lengths)

            padded = np.zeros((len(batch), max_len, 2), dtype=np.float32)
            padded_times = np.zeros((len(batch), max_len, 1), dtype=np.float32)
            masks = np.zeros((len(batch), max_len), dtype=np.float32)
            for i, item in enumerate(batch):
                seq_len = item['cgyro'].shape[0]
                padded[i, :seq_len, :] = item['cgyro']
                padded_times[i, :seq_len, :] = item['time']
                masks[i, :seq_len] = 1.0

            result = {
                'cgyro': padded,          # (B, max_len, 2)
                'time': padded_times,     # (B, max_len, 1)
                'mask': masks,            # (B, max_len)
                'lengths': np.array(lengths),
            }

    # Add per-sample normalization stats if present
    if has_sample_stats:
        result['sample_mean'] = np.stack([item['sample_mean'] for item in batch], axis=0)  # (B, 1, 2)
        result['sample_std'] = np.stack([item['sample_std'] for item in batch], axis=0)    # (B, 1, 2)

    return result


def create_cgyro_dataloader(
    dataset: CGYRODataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for CGYRO data.
    
    Note: For variable-length sequences, batch_size=1 is recommended.
    Larger batches require padding and careful handling.
    
    Args:
        dataset: CGYRODataset instance
        batch_size: Batch size (recommend 1 for variable length)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader instance
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,  # Keep all samples for variable length
        collate_fn=cgyro_collate_fn,
    )
    return loader


# =============================================================================
# Paired Dataset (for diffusion training)
# =============================================================================

class PairedDataset(Dataset):
    """Dataset that returns paired (TGLF, CGYRO) samples.
    
    Used for diffusion training where we need both conditioning (TGLF)
    and target (CGYRO) data.
    
    For the new structured layout, data_path should point to:
    /home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields
    
    All folders in 2species_2fields are guaranteed to have both TGLF and CGYRO data.
    The dataset now contains ~253 folders (increased from ~154).
    
    Args:
        data_path: Path to directory containing run folders (should be 2species_2fields)
        normalize_tglf: Whether to normalize TGLF data
        normalize_cgyro: Whether to normalize CGYRO data
    """
    
    def __init__(
        self,
        data_path: str,
        normalize_tglf: bool = False,
        normalize_cgyro: bool = False,
    ):
        self.data_path = Path(data_path)
        
        # Get paired folders
        self.folders = get_paired_folders(data_path)
        
        if len(self.folders) == 0:
            raise ValueError(f"No valid folders found in {data_path}")
        
        print(f"PairedDataset: Found {len(self.folders)} paired samples")
        
        # Load TGLF data
        self.tglf_data = []
        for folder in self.folders:
            npz_path = folder / "conditioning_data.npz"
            sample = load_tglf_sample(npz_path)
            self.tglf_data.append(sample)
        self.tglf_data = np.stack(self.tglf_data, axis=0)  # (N, 21, 108)
        
        # Load CGYRO data (variable length)
        self.cgyro_data = []
        self.cgyro_lengths = []
        for folder in self.folders:
            h5_path = find_h5_file(folder)
            sample = load_cgyro_sample(h5_path)
            self.cgyro_data.append(sample)
            self.cgyro_lengths.append(sample.shape[0])
        self.cgyro_lengths = np.array(self.cgyro_lengths)
        
        # Normalization
        if normalize_tglf:
            self.tglf_mean = self.tglf_data.mean(axis=(0, 1), keepdims=True)
            self.tglf_std = self.tglf_data.std(axis=(0, 1), keepdims=True) + 1e-8
            self.tglf_data = (self.tglf_data - self.tglf_mean) / self.tglf_std
        else:
            self.tglf_mean = None
            self.tglf_std = None
        
        if normalize_cgyro:
            all_cgyro = np.concatenate(self.cgyro_data, axis=0)
            self.cgyro_mean = all_cgyro.mean(axis=0, keepdims=True)
            self.cgyro_std = all_cgyro.std(axis=0, keepdims=True) + 1e-8
            for i in range(len(self.cgyro_data)):
                self.cgyro_data[i] = (self.cgyro_data[i] - self.cgyro_mean) / self.cgyro_std
        else:
            self.cgyro_mean = None
            self.cgyro_std = None
    
    def __len__(self) -> int:
        return len(self.folders)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            'tglf': self.tglf_data[idx],      # (21, 108)
            'cgyro': self.cgyro_data[idx],    # (n_timesteps, 2) - variable!
            'cgyro_length': self.cgyro_lengths[idx],
        }
    
    def get_folder_name(self, idx: int) -> str:
        """Get the folder name for a given index."""
        return self.folders[idx].name

