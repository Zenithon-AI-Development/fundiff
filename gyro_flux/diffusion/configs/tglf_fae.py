import ml_collections

from gyro_flux.diffusion.configs import models


def get_config(model="tglf_fae"):
    """Get the hyperparameter configuration for TGLF FAE training."""
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
    # TGLF FAE: Fixed (B, 21, 108) - 21 ky modes, 108 features
    # Encoder outputs z_c (B, z_c_dim) directly for DiT conditioning
    config.num_ky_modes = 21             # Fixed number of ky modes
    config.num_features = 108            # Fixed features per mode
    config.x_dim = [2, 21, 108]          # [dummy_batch=2, ky_modes, features]
    config.ky_query_dim = [2, 21, 1]     # [dummy_batch=2, num_queries, 1] for ky queries
    # NOTE: z_c_aggregation, z_c_dim are in config.model.encoder (from models.py)

    # Training or evaluation
    config.mode = "train_tglf_fae"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.use_wandb = True               # Set True to enable W&B logging
    wandb.project = "gyro_flux"
    wandb.entity = "Zenithon-AI"
    wandb.group = "week19jan"
    wandb.run_name = "tglf-fae-v2-small"
    wandb.notes = "v2-small: Reduced model (emb=128, latents=8, depth=2) to prevent overfitting on ~180 samples."
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/home/shared_info/Well_Formatted_CGYRO_W_TGLF_structured/2species_2fields/"
    dataset.num_train_samples = None     # Number of training samples (None = all)
    dataset.train_batch_size = 32        # ~8 batches/epoch with ~253 samples
    dataset.test_batch_size = 32
    dataset.num_workers = 4

    # Learning rate schedule (cosine decay with warmup)
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 3e-4                 # Conservative for small dataset
    lr.decay_rate = 0.1                  # Decay to 10% of peak
    lr.transition_steps = 10000          # Decay over this many steps
    lr.warmup_steps = 500                # ~62 epochs warmup (253/32 ≈ 8 batches/epoch)

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-3            # Higher regularization for small dataset
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 10_000          # ~1250 epochs (253/32 ≈ 8 batches/epoch)
    training.num_queries = 21            # Reconstruct ALL 21 ky modes each step

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 100           # Every 100 steps (~12-13 epochs)

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 1000          # Every 1000 steps (~125 epochs)
    saving.num_keep_ckpts = 5

    return config

