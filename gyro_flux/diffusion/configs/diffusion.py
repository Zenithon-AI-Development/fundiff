import ml_collections

from configs import models


def get_config(model_pair="target_fae,flux_dit"):
    """Get the hyperparameter configuration for diffusion training.
    
    Args:
        model_pair: Comma-separated string of "autoencoder,diffusion" model names.
                   e.g., "target_fae,flux_dit"
    """
    config = get_base_config()

    # Parse model pair
    autoencoder_name, diffusion_name = model_pair.split(',')
    
    get_autoencoder_config = getattr(models, f"get_{autoencoder_name}_config")
    get_diffusion_config = getattr(models, f"get_{diffusion_name}_config")

    config.autoencoder = get_autoencoder_config()
    config.diffusion = get_diffusion_config()
    
    # Also need TGLF encoder for conditioning
    config.tglf_encoder = models.get_tglf_fae_config()
    
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shapes for initializing Flax models (dummy batch size for param init)
    # z: latent from Target FAE (B, num_latents, emb_dim)
    config.z_dim = [2, 64, 256]          # [dummy_batch=2, num_latents, emb_dim]
    config.c_dim = [2, 256]              # [dummy_batch=2, cond_dim] - TGLF condition vector
    config.t_dim = [2,]                  # [dummy_batch=2,] - diffusion timestep

    # Training or evaluation
    config.mode = "train_diffusion"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "gyro_flux"
    wandb.entity = None
    wandb.run_name = None
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = None             # Path to paired TGLF/CGYRO data
    dataset.num_train_samples = None
    dataset.batch_size = 128             # Per device (latent space, can use larger batch)
    dataset.test_batch_size = 64
    dataset.num_workers = 4

    # Pretrained FAE checkpoints
    config.checkpoints = ckpts = ml_collections.ConfigDict()
    ckpts.target_fae_path = None         # Path to trained Target FAE checkpoint
    ckpts.tglf_fae_path = None           # Path to trained TGLF FAE checkpoint

    # Learning rate schedule
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-4                 # Lower LR for diffusion
    lr.decay_rate = 0.95
    lr.transition_steps = 10000
    lr.warmup_steps = 2000

    # Optimizer (AdamW)
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-4
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 200_000         # Diffusion typically needs more steps

    # Evaluation / Sampling
    config.eval = eval = ml_collections.ConfigDict()
    eval.num_samples = 64                # Number of samples to generate for eval
    eval.num_steps = 100                 # ODE integration steps for sampling
    eval.batch_size = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 100

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 1000
    saving.num_keep_ckpts = 5

    return config

