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
    # Target FAE: (B, T, C) where T varies ~150-3000, using 1000 as example
    config.x_dim = [2, 1000, 2]          # [dummy_batch=2, example_timesteps, channels]
    config.step_query_dim = [2, 256]     # [dummy_batch=2, num_queries] integer step indices

    # Training or evaluation
    config.mode = "train_target_fae"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.use_wandb = True              # Set True to enable W&B logging
    wandb.project = "gyro_flux"
    wandb.entity = "Zenithon-AI"
    wandb.group = "target_fae"
    wandb.run_name = "target-fae-test4_time"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF/"
    dataset.num_train_samples = None     # Number of training samples (None = all)
    dataset.train_batch_size = 16         # 1 for variable-length, can increase for padded
    dataset.test_batch_size = 1
    dataset.num_workers = 4
    
    # Padding options for efficient compilation
    dataset.use_padded_sequences = True  # If True, pad all sequences to max_seq_length
    dataset.max_seq_length = 3000         # Max sequence length for padding

    # Learning rate schedule
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 3e-4
    lr.decay_rate = 0.9
    lr.transition_steps = 20000
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
    training.max_steps = 20000
    training.num_queries = 256           # Number of random time points to sample per batch
    training.use_time_weighting = True  # Weight early timesteps (growth phase) more
    training.growth_phase_weight = 5.0   # Weight multiplier for t < 0.02 (~24 steps)

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 50

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 500
    saving.num_keep_ckpts = 3

    return config

