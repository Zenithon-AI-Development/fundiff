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
    """
    
    def __init__(
        self,
        data_path: str,
        paired_only: bool = True,
        normalize: bool = False,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        self.data_path = Path(data_path)
        self.paired_only = paired_only
        self.normalize = normalize
        
        # Get folder list
        if paired_only:
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
        flux_species0 = f['fluxes/total_flux_species0'][0, :, 1]
        flux_species1 = f['fluxes/total_flux_species1'][0, :, 1]
        flux = np.stack([flux_species0, flux_species1], axis=1)
        
        times = f['times'][:]
    
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
        normalize: If True, apply normalization
        stats: Optional (mean, std) for normalization
        use_padding: If True, pad all sequences to max_seq_length for efficient JAX compilation
        max_seq_length: Maximum sequence length for padding (ignored if use_padding=False)
    """
    
    def __init__(
        self,
        data_path: str,
        normalize: bool = False,
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_padding: bool = False,
        max_seq_length: int = 3000,
    ):
        self.data_path = Path(data_path)
        self.normalize = normalize
        self.use_padding = use_padding
        self.max_seq_length = max_seq_length
        
        # Only use paired folders (those with both TGLF and CGYRO)
        self.folders = get_paired_folders(data_path)
        
        if len(self.folders) == 0:
            raise ValueError(f"No valid folders found in {data_path}")
        
        print(f"CGYRODataset: Found {len(self.folders)} samples")
        
        # Load all data into memory (variable length sequences)
        self.data = []
        self.lengths = []
        
        for folder in self.folders:
            h5_path = find_h5_file(folder)
            if h5_path is None:
                raise ValueError(f"No H5 file found in {folder}")
            
            sample = load_cgyro_sample(h5_path)  # (n_timesteps, 2)
            self.data.append(sample)
            self.lengths.append(sample.shape[0])
        
        self.lengths = np.array(self.lengths)
        print(f"CGYRODataset: Timestep range [{self.lengths.min()}, {self.lengths.max()}]")
        
        # Normalization (per-channel)
        if normalize:
            if stats is not None:
                self.mean, self.std = stats
            else:
                # Compute stats across all samples and timesteps
                all_data = np.concatenate(self.data, axis=0)  # (total_timesteps, 2)
                self.mean = all_data.mean(axis=0, keepdims=True)  # (1, 2)
                self.std = all_data.std(axis=0, keepdims=True) + 1e-8  # (1, 2)
            
            # Apply normalization
            for i in range(len(self.data)):
                self.data[i] = (self.data[i] - self.mean) / self.std
        else:
            self.mean = None
            self.std = None
        
        # If using padding, pad all sequences now
        if use_padding:
            print(f"CGYRODataset: Padding all sequences to {max_seq_length}")
            self.padded_data = np.zeros((len(self.data), max_seq_length, 2), dtype=np.float32)
            self.masks = np.zeros((len(self.data), max_seq_length), dtype=np.float32)
            
            for i, sample in enumerate(self.data):
                seq_len = min(sample.shape[0], max_seq_length)
                self.padded_data[i, :seq_len, :] = sample[:seq_len]
                self.masks[i, :seq_len] = 1.0
    
    def __len__(self) -> int:
        return len(self.folders)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.use_padding:
            return {
                'cgyro': self.padded_data[idx],  # (max_seq_length, 2)
                'mask': self.masks[idx],          # (max_seq_length,)
                'length': self.lengths[idx],      # original length
            }
        else:
            return {
                'cgyro': self.data[idx],          # (n_timesteps, 2) - variable!
                'length': self.lengths[idx],
            }
    
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
    
    Args:
        batch: List of dicts from CGYRODataset.__getitem__
    
    Returns:
        Dictionary with 'cgyro', 'lengths', and optionally 'mask' keys
    """
    # Check if we're in padded mode
    has_mask = 'mask' in batch[0]
    
    if has_mask:
        # Padded mode: all sequences are same length
        cgyro = np.stack([item['cgyro'] for item in batch], axis=0)
        masks = np.stack([item['mask'] for item in batch], axis=0)
        lengths = np.array([item['length'] for item in batch])
        return {
            'cgyro': cgyro,      # (B, max_seq_length, 2)
            'mask': masks,       # (B, max_seq_length)
            'lengths': lengths,  # (B,)
        }
    else:
        # Variable-length mode
        if len(batch) == 1:
            return {
                'cgyro': batch[0]['cgyro'][None, ...],  # (1, n_timesteps, 2)
                'lengths': np.array([batch[0]['length']]),
            }
        
        # For batch_size > 1, pad to max length in batch
        lengths = [item['length'] for item in batch]
        max_len = max(lengths)
        
        padded = np.zeros((len(batch), max_len, 2), dtype=np.float32)
        masks = np.zeros((len(batch), max_len), dtype=np.float32)
        for i, item in enumerate(batch):
            seq_len = item['cgyro'].shape[0]
            padded[i, :seq_len, :] = item['cgyro']
            masks[i, :seq_len] = 1.0
        
        return {
            'cgyro': padded,      # (B, max_len, 2)
            'mask': masks,        # (B, max_len)
            'lengths': np.array(lengths),
        }


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

