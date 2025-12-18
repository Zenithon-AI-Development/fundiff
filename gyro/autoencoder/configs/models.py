import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS (same pattern as Burgers)."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_gyro_fae_fno_config():
    """Gyro FAE: FNO encoder + coordinate-conditioned cross-attn decoder."""
    config = ml_collections.ConfigDict()
    config.model_name = "GYRO_FAE_FNO"

    config.encoder = enc = ml_collections.ConfigDict()

    # Data layout assumptions (only used by the model; dataloader can be channels-first or last)
    enc.in_channels = 4  # RePhi, ImPhi, ReApar, ImApar

    # Latent shape: (B, num_latents, latent_dim)
    enc.num_latents = 64
    enc.latent_dim = 128

    # FNO backbone
    enc.fno_emb_dim = 128
    enc.fno_depth = 4
    enc.modes1 = 12
    enc.modes2 = 12
    enc.fno_padding = 0

    # Perceiver bottleneck
    enc.perceiver_depth = 2
    enc.num_heads = 8
    enc.mlp_ratio = 1
    enc.layer_norm_eps = 1e-5

    # Decoder: coords -> cross-attn into latents -> MLP head
    config.decoder = dec = ml_collections.ConfigDict()
    dec.enabled = True

    # Dimensions
    dec.latent_dim = enc.latent_dim
    dec.dec_emb_dim = 128
    dec.out_dim = 4

    # Fourier feature mapping for coords (kx, ky)
    dec.fourier_freq = 10.0

    # Cross-attention stack
    dec.dec_depth = 4
    dec.dec_num_heads = 8
    dec.mlp_ratio = 2
    dec.layer_norm_eps = 1e-5

    # Projection head
    dec.num_mlp_layers = 2

    # Physics constraint (conjugate symmetry)
    dec.enforce_conjugate_symmetry = True
    dec.zero_tol = 0.0

    return config


