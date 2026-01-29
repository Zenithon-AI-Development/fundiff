# Project Context: Gyrokinetic Flux Prediction with Diffusion Models

## High-Level Goal

This project trains diffusion models to generate **gyrokinetic flux time-series** (energy flux from ion and electron species) conditioned on **TGLF stability analysis vectors**. The goal is to learn a generative model that can predict flux evolution trajectories given initial stability conditions, enabling faster plasma turbulence simulations.

**Problem Statement:** Traditional gyrokinetic simulations (CGYRO) are computationally expensive. We want to learn a generative model that can produce realistic flux trajectories from cheaper TGLF stability analysis, enabling rapid exploration of parameter spaces.

## Architecture Overview

The system uses a **two-stage training pipeline**:

### Stage 1: Function Autoencoders (FAEs)
Two separate FAEs compress different types of data into latent representations:

1. **TGLF FAE** (`gyro_flux/models/tglf_fae.py`):
   - Encodes TGLF stability spectra `(21, 108)` → conditioning vector `z_c` `(256,)`
   - Fixed input shape, simpler architecture
   - Used as conditioning for diffusion model

2. **Target FAE** (`gyro_flux/models/time_fae.py`):
   - Encodes CGYRO flux time-series `(T, 2)` + time coordinates (T, 1) → latent `z_1` `(128, 256)`
   - Variable-length input sequences (T varies ~150-3000 timesteps)
   - **Index-invariant architecture** (see Key Decisions below)
   - This is what we're currently training

### Stage 2: Diffusion Model (FluxDiT)
- **Rectified Flow** diffusion in latent space
- Generates `z_1` (Target FAE latents) from noise, conditioned on `z_c` (TGLF latents)
- Uses **DiT (Diffusion Transformer)** architecture with AdaLN-Zero modulation
- Frozen FAE encoders during diffusion training

## Key Architectural Decisions

### 1. Index-Invariant Architecture (Recent Major Refactoring)

**Problem:** Original model used array indices (positional embeddings), which is physically wrong. Two files with different time densities would map the same physical time to different indices.

**Solution:** Model now uses **explicit physical time coordinates** instead of indices.

**Implementation:**
- **Time Normalization:** `τ = (t - 3.0) / 1000.0`
  - `3.0` is global phase cutoff (all simulations start at t=3.0 or later)
  - `1000.0` is time scale for numerical stability
  - Preserves physics: growth phase always at consistent `τ` regardless of file start time

- **Encoder (`TimeSeriesEncoder`):**
  - Input: `x` (flux `(B, T, 2)`) + `t` (normalized time `(B, T, 1)`)
  - Pointwise flux embedding: `Dense(emb_dim)` (no patching, preserves temporal resolution)
  - Fourier time embedding: `FourierEmbs(freq=150)` applied to `t`
  - Fusion: element-wise addition of flux + time embeddings
  - Perceiver bottleneck: variable T → fixed `num_latents=128`
  - Self-attention transformer blocks

- **Decoder (`ContinuousTimeDecoder`):**
  - Input: `z` (latent `(B, 128, 256)`) + `t_query` (continuous time queries `(B, N_q, 1)`)
  - Uses same Fourier frequency (150) as encoder for consistency
  - Cross-attention: queries attend to latent representation
  - Outputs flux at arbitrary time points

**Key Files:**
- `gyro_flux/models/time_fae.py`: Model architecture
- `gyro_flux/data_utils.py`: Time normalization in `CGYRODataset`
- `gyro_flux/diffusion/model_utils.py`: Loss functions and batch preparation

### 2. Variable-Length Sequence Handling

**Three modes** (controlled by config):

1. **Slice Mode** (`slice_some_time=True`):
   - Takes first `slice_length` timesteps from each sequence
   - All sequences become fixed-length → can use larger batch sizes
   - Currently: `slice_length=50`, `batch_size=32`

2. **Padding Mode** (`use_padded_sequences=True`):
   - Pads all sequences to `max_seq_length`
   - Fixed batch size, but wastes computation on padding

3. **Variable-Length Mode** (both False):
   - True variable-length sequences
   - Must use `batch_size=1` (JAX limitation)
   - Most flexible but slower

**Current Training:** Using slice mode (50 timesteps) for faster iteration.

### 3. Data Normalization

- **Global Z-score normalization** computed from training set only
- Applied to all splits (train/val/test) using same statistics
- Prevents data leakage from validation/test sets

### 4. Training Strategy

- **80/10/10 train/val/test split** with fixed seed for reproducibility
- Validation runs at end of each epoch
- Logs `loss/train` and `loss/val` to W&B
- Learning rate: warmup to `3e-4`, then cosine decay
- Gradient clipping: `clip_norm=1.0`

## Domain-Specific Concepts

### Gyrokinetics
- **CGYRO**: High-fidelity gyrokinetic simulation code
  - Produces flux time-series: energy flux for ions and electrons over time
  - Output stored in H5 files: `t0_fields/total_flux_species0` (ions), `species1` (electrons)
  - Flux shape: `(1, T, 3)` where last dim is [particle, energy, momentum] flux
  - We use **energy flux** (index 1) only
  - Time values stored in `times` dataset: `(T,)` float32 array

- **TGLF**: Faster linear stability analysis code
  - Produces stability spectra: growth rates, frequencies, flux spectra
  - Output stored in `conditioning_data.npz`
  - Shape: `(21, 108)` after preprocessing
  - Used as conditioning for diffusion model

### Data Structure
- **Paired folders**: Each simulation run has both TGLF and CGYRO outputs
- **H5 files**: Named same as folder (e.g., `2020_08-nuei_lor_alor6_filtered.h5`)
- **Time inconsistencies**: Files start at different times (3.0, 3.06, 3.15, 3.2, 3.6, 4.0)
  - Handled by global phase cutoff normalization
- **Variable lengths**: Sequences range from ~150 to ~3000 timesteps

### Physics Context
- **Growth phase**: Early time period (τ < 0.03) where instabilities grow
  - Can optionally weight this phase more heavily in loss (`use_time_weighting=True`)
- **Energy flux**: Non-negative quantity (model uses Softplus output activation)
- **Chaotic dynamics**: Flux trajectories are highly chaotic, making prediction difficult

## Current Status

### The current (last run) approach
- ✅ Index-invariant architecture implemented and training
- ✅ Time coordinate normalization (`τ = (t - 3.0) / 1000.0`)
- ✅ Fourier embeddings for time (freq=150)
- ✅ Validation split and logging to W&B
- ✅ Slice mode training (50 timesteps, batch_size=32)

### Current Training Run
- **Model**: Target FAE (trel-v1)
- **Config**: `fourier_freq=150`, `num_latents=128`, `emb_dim=256`, `depth=8`
- **Data**: 227 folders (181 train, 23 val, 23 test)
- **Status**: Training loss decreasing (~0.05-0.07), validation loss higher (~0.3-0.6) with spikes
- **W&B Project**: `gyro_flux_target_training`, run `target-fae-trel-v1`

### Known Issues / Observations
1. **Validation loss spikes**: Occasional large spikes, instability in training
2. **Train/val gap**: Validation loss consistently higher, indicating some overfitting
3. **Training instability**: Loss curves show variance, may need longer training or different hyperparameters

## Gotchas and Non-Obvious Things

### 1. Time Coordinate Normalization
- **Critical**: Never subtract `times[0]` (file-specific start time). Always subtract global `cutoff=3.0`.
- **Why**: Preserves physical phase information. Files starting at 4.0 should have different `τ` than those starting at 3.0.

### 3. Fourier Frequency Consistency
- **Critical**: Encoder and decoder must use **same** `fourier_freq` (currently 150)
- **Why**: Ensures "Time 0.5" looks the same to both encoder and decoder
- **Location**: `gyro_flux/diffusion/configs/models.py` - both encoder and decoder configs

### 4. Batch Preparation
- **Training**: `prepare_target_batch()` samples random indices, then extracts corresponding time values
- **Validation**: Can use `prepare_target_batch_arbitrary_times()` for custom time queries
- **Location**: `gyro_flux/diffusion/model_utils.py`

### 5. Data Filtering
- **Excluded folders**: Any folder with `"2023_04-exb_paper"` in name is skipped
- **Why**: These files have different characteristics (negative flux values, different time structure)
- **Location**: `gyro_flux/data_utils.py:get_paired_folders()`

### 6. Sequence Length Handling
- **Slice mode**: If `slice_length > min_sequence_length`, automatically adjusts to minimum
- **Location**: `gyro_flux/data_utils.py:CGYRODataset.__init__()`

### 7. Padding vs Masking
- **Current**: Padding uses zero values, model learns to ignore them
- **Future**: Could add explicit attention masks for better handling

### 8. Normalization Statistics
- **Computed once** from training set only
- **Applied globally** to all splits
- **Location**: `gyro_flux/diffusion/train_target_fae.py` - normalization stats computed before split

### 9. RNG Keys for Reproducibility
- **Data split**: Uses `random_state=config.seed` in `train_test_split`
- **Validation**: Uses `config.seed + epoch` for validation batch sampling
- **Training**: Uses separate RNG key that's split each iteration

### 10. Diffusion Training (Future)
- **Frozen encoders**: Both TGLF and Target FAE encoders are frozen during diffusion training
- **Rectified Flow**: Uses velocity prediction, not noise prediction
- **Location**: `gyro_flux/diffusion/train_diffusion.py` (not yet fully updated for new architecture)

## File Structure

```
gyro_flux/
├── models/
│   ├── time_fae.py          # Target FAE (TimeSeriesEncoder, ContinuousTimeDecoder)
│   ├── tglf_fae.py          # TGLF FAE (TGLFEncoder, TGLFDecoder)
│   └── flux_dit.py          # Diffusion Transformer (FluxDiT)
├── data_utils.py            # Data loading, CGYRODataset, time normalization
├── diffusion/
│   ├── train_target_fae.py  # Target FAE training script (current focus)
│   ├── train_tglf_fae.py   # TGLF FAE training script
│   ├── train_diffusion.py  # Diffusion model training (Stage 2)
│   ├── model_utils.py      # Loss functions, train steps, batch preparation
│   └── configs/
│       ├── target_fae.py   # Target FAE hyperparameters
│       ├── tglf_fae.py     # TGLF FAE hyperparameters
│       ├── diffusion.py    # Diffusion model hyperparameters
│       └── models.py       # Model architecture configs
└── utils/
    ├── inspect_h5.py        # H5 file inspection utilities
    └── plot_fluxes.py       # Flux plotting utilities
```

## References

- **FunDiff Paper**: Diffusion Models over Function Spaces
- **CGYRO**: Gyrokinetic simulation code
- **TGLF**: Trapped Gyro-Landau Fluid code
- **Rectified Flow**: Straight path diffusion models
- **Perceiver**: Variable-length sequence processing architecture
