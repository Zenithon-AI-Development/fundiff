import ml_collections

from configs import models


def get_config(model: str):
    """Get the hyperparameter configuration for a specific gyro autoencoder model.

    Mirrors `burgers/diffusion/configs/autoencoder.py`.
    """
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Default gyro autoencoder config (encoder-only for now)."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Shapes for initializing Flax models
    # NOTE: Gyro encoder supports channels-first (B, 4, Nx, Ny) OR channels-last (B, Nx, Ny, 4).
    # Pick a small default here just for init; training can use different resolutions.
    config.x_dim = [2, 4, 64, 64]

    # Training / evaluation mode (kept for parity with burgers)
    config.mode = "train_autoencoder"

    # Weights & Biases (optional)
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_gyro_example"
    wandb.entity = None
    wandb.run_name = None
    wandb.tag = None

    # Dataset (placeholders; wire up your gyro dataset later)
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = None  # e.g. "path/to/gyro_dataset"
    dataset.train_batch_size = 4  # per device
    dataset.test_batch_size = 1   # per device
    dataset.num_workers = 8

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 2000
    lr.warmup_steps = 2000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 1_000
    training.random_resolution = True
    training.num_queries = -1  # -1 = full grid, otherwise randomly sample that many coords per snapshot

    # Losses (gyro physics-aware autoencoder)
    training.loss = loss = ml_collections.ConfigDict()

    loss.rec = rec = ml_collections.ConfigDict()
    rec.enabled = True
    rec.weight = 1.0
    rec.kind = "l1"  # "l1", "huber", "mse"
    rec.huber_delta = 1.0

    loss.spec = spec = ml_collections.ConfigDict()
    spec.enabled = True
    spec.weight = 0.1
    spec.num_bins = 128
    spec.eps = 1e-8

    loss.cross = cross = ml_collections.ConfigDict()
    cross.enabled = True
    cross.weight = 1.0

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 1

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 50
    saving.num_keep_ckpts = 2

    return config


