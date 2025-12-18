import ml_collections

from configs import models


def get_config(model):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models
    config.x_dim = [2, 200, 200, 1]
    config.coords_dim = [2,]  # Only for initializing CViT model

    # Training or evaluation
    config.mode = "train_autoencoder"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_burgers_example"
    wandb.entity = "Zenithon-AI"  # Set to your W&B entity (e.g. "guzmans") if needed
    wandb.run_name = None  # Optional explicit run name; falls back to job_name if None
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    # Path is relative to the training script working directory (`burgers/diffusion`).
    # Expect the dataset at `burgers/data/burger_nu_1e-3.mat` inside the repo.
    dataset.data_path = "../data/burger_nu_1e-3.mat"
    dataset.downsample_factor = 1
    dataset.num_train_samples = 3600
    dataset.train_batch_size = 16  # Per device
    dataset.test_batch_size = 4  # Per device
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
    # Number of gradient steps; ~225 steps ≈ 1 epoch for the default Burgers dataset,
    # so 1_000 steps is roughly 4–5 epochs for a quick smoke test.
    # training.max_steps = 1 * 10**5  previous setup
    training.max_steps = 1_000
    training.num_queries = 4096
    training.random_resolution = True
    training.use_pde = False

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 1

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 2
    saving.num_keep_ckpts = 1

    return config
