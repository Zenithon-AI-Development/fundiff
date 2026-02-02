import ml_collections

from gyro_flux.diffusion.configs import models


def get_config(model="target_fae"):
    """Get the hyperparameter configuration for Target FAE training."""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models (dummy batch size for param init)
    # Target FAE: (B, T, C) where T=256 after slicing
    config.x_dim = [2, 256, 2]           # [dummy_batch=2, slice_length, channels] flux input
    config.t_dim = [2, 256, 1]           # [dummy_batch=2, slice_length, 1] normalized time coordinates
    config.t_query_dim = [2, 64, 1]      # [dummy_batch=2, num_queries, 1] normalized time queries

    # Training or evaluation
    config.mode = "train_target_fae"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.use_wandb = True              # Set True to enable W&B logging
    wandb.project = "gyro_flux_target_training"
    wandb.entity = "Zenithon-AI"
    wandb.group = "week26jan"
    wandb.run_name = "target-fae-skip-perceiver-film-v1"
    wandb.notes = "SKIP-PERCEIVER: Full self-attention with explicit masking, FiLM physics conditioning. Tests if Perceiver bottleneck is lossy."
    wandb.tags = ["target_fae", "skip_perceiver", "film_conditioning", "global_norm"]

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields/"
    dataset.num_train_samples = None     # Number of training samples (None = all)
    dataset.train_batch_size = 32        # Increased for more stable gradients
    dataset.test_batch_size = 32
    dataset.num_workers = 4

    # Padding options for efficient compilation
    dataset.use_padded_sequences = True   # Required for time-based slicing (variable lengths after slicing)
    dataset.max_seq_length = 3000         # Fallback max length (overridden by time-based slicing)

    # Slice strategy: use time-based slicing for consistent physics coverage
    # OLD (deprecated): slice_some_time slices by array index - inconsistent across different dt
    dataset.slice_some_time = False      # Deprecated: slices by index, not physical time
    dataset.slice_length = 256           # Fallback if using index-based slicing

    # NEW: Time-based slicing ensures consistent physical time coverage across all samples
    # τ = (t - 3.0) / 1000.0, so max_tau=0.1 means t ∈ [3.0, 103.0] (100 time units)
    dataset.slice_by_time = True         # Use physical time instead of array indices
    dataset.max_tau = 0.1                # Max normalized time to include (100 time units of physics)

    # Random time window augmentation (NEW)
    # PREVIOUS: Always used fixed slice starting from τ=0.
    # CURRENT: Randomly samples start point τ_start ∈ [0, τ_max - window_size] per sample.
    # This prevents overfitting to early-phase patterns and increases sample diversity.
    dataset.use_random_window = True     # Enabled for random window augmentation
    dataset.window_tau_size = 0.1        # Window size in τ units (matches max_tau)

    # Parameter space filtering
    # Option 1: Substring matching (original)
    dataset.use_param_space = False      # Set True to filter by parameter space substring
    dataset.param_space_name = ""        # Substring to match in folder names
    dataset.excluded_runs = []           # Specific runs to exclude
    
    # Option 2: CSV-based filtering (NEW - preferred for validated param spaces)
    dataset.use_param_list_csv = True    # Set True to use CSV for folder filtering
    dataset.param_list_csv_path = "gyro_flux/validated_parameters.csv"  # Relative path (CSV bundled with code)
    
    # Physics conditioning (NEW)
    # Inject plasma physics parameters as conditioning token in encoder
    dataset.use_physics_conditioning = True    # ENABLED: Skip-Perceiver with FiLM
    dataset.physics_param_columns = ["DLNTDR_1", "DLNNDR_1", "KY", "NU_EE", "MASS_1"]  # 5 varying params

    # Normalization: addresses 324x magnitude variation across files
    # PHASE 1 EXPERIMENT: Global norm preserves magnitude info for physics conditioning
    dataset.normalize_per_sample = False  # DISABLED: Use global norm instead
    dataset.normalize_global = True       # ENABLED: Preserves inter-sample magnitude differences

    # Learning rate schedule
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 3e-4
    lr.decay_rate = 0.9
    lr.transition_steps = 30000          # Match max_steps for proper decay schedule
    lr.warmup_steps = 1000

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-4
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 30000           # Extended training for better convergence
    training.num_queries = 64            # Number of random time points to sample per batch (reduced for smaller model)
    training.use_time_weighting = False  # Weight early timesteps (growth phase) more
    training.growth_phase_weight = 5.0

    # Loss function: relative L2 for scale-invariance
    # NOTE: Disabled - redundant with per-sample normalization. After per-sample norm,
    # all samples have similar target² values, so rel L2 weights become constant (~0.5)
    training.use_relative_l2 = False     # Disabled (redundant with per-sample normalization)
    training.rel_l2_eps = 1.0            # Epsilon for denominator stability (unused when disabled)   

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 10  # Frequent logging for diagnostic correlation analysis

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 500
    saving.num_keep_ckpts = 3

    return config

