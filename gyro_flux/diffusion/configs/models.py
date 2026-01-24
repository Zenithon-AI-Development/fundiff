import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_target_fae_config():
    """Configuration for the CGYRO time-series FAE (Target History FAE).

    Index-invariant architecture using explicit physical time coordinates.
    Variable-length input: (B, T, 2) + (B, T, 1) time where T ranges ~150 to ~3000 timesteps
    Time coordinate: τ = (t - 3.0) / 1000.0 (normalized relative to global phase cutoff)
    Latent: (B, num_latents, emb_dim)
    Output: (B, N_queries, 2) at arbitrary timesteps (via continuous time queries)

    Model sizing rationale (v2):
    - slice_length=256 gives 8:1 compression to num_latents=32
    - Smaller emb_dim=128 matches reduced data per sample
    - Shallower depth=4 prevents overfitting on ~180 samples
    """
    config = ml_collections.ConfigDict()
    config.model_name = "TARGET_FAE"

    # Encoder: (B, T, 2) + (B, T, 1) -> (B, num_latents, emb_dim), T varies
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.in_channels = 2              # Q_i, Q_e
    encoder.emb_dim = 128                # Reduced: 256 -> 128 (smaller model for limited data)
    encoder.num_latents = 32             # Reduced: 128 -> 32 (8:1 compression from 256 timesteps)
    encoder.perceiver_depth = 2          # Cross-attn layers in Perceiver bottleneck
    encoder.transformer_depth = 4        # Reduced: 8 -> 4 (prevent overfitting)
    encoder.num_heads = 4                # Reduced: 8 -> 4 (proportional to emb_dim)
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5
    encoder.fourier_freq = 150.0         # Must match decoder - Fourier frequency for time embedding

    # Decoder: (B, num_latents, emb_dim) + t_query -> (B, N_queries, 2)
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.latent_dim = 128             # Must match encoder.emb_dim
    decoder.dec_emb_dim = 128            # Reduced: 256 -> 128
    decoder.dec_depth = 4                # Reduced: 8 -> 4
    decoder.dec_num_heads = 4            # Reduced: 8 -> 4
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2           # MLP layers in output head
    decoder.out_channels = 2             # Q_i, Q_e
    decoder.fourier_freq = 150.0         # Must match encoder - Fourier frequency for time embedding
    decoder.use_softplus = True          # Non-negativity constraint for heat flux
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_tglf_fae_config():
    """Configuration for the TGLF conditioning FAE.

    Fixed input: (B, 21, 108) - 21 ky modes, 108 features per mode
    Features: 80 ql_weights + 4 gamma + 4 freq + 20 flux predictions = 108
    Latent: z_c (B, z_c_dim) - directly usable for DiT conditioning
    Output: (B, N_queries, 108) at queried ky positions

    Model sizing rationale (v2-small):
    - ~180 train samples with 21×108=2268 values each
    - Reduced emb_dim=128, num_latents=8, depth=2 to prevent overfitting
    - z_c_dim=128 (will need DiT config update to match)
    """
    config = ml_collections.ConfigDict()
    config.model_name = "TGLF_FAE"

    # Encoder: (B, 21, 108) -> z_c (B, z_c_dim)
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.num_ky_modes = 21            # Fixed number of ky modes
    encoder.input_dim = 108              # Fixed features per mode
    encoder.emb_dim = 128                # Reduced: 256 -> 128
    encoder.num_latents = 8              # Reduced: 16 -> 8 (2.6:1 compression from 21 ky modes)
    encoder.perceiver_depth = 1          # Reduced: 2 -> 1
    encoder.transformer_depth = 2        # Reduced: 4 -> 2
    encoder.num_heads = 4                # Reduced: 8 -> 4
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5
    # Aggregation to z_c
    encoder.z_c_aggregation = "mean"     # "mean" or "flatten"
    encoder.z_c_dim = 128                # Reduced: 256 -> 128
    encoder.use_z_c_projection = False   # No projection - mean pooling gives (B, 128) directly

    # Decoder: z_c (B, z_c_dim) + ky_query -> (B, N_queries, 108)
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.z_c_dim = 128                # Must match encoder.z_c_dim
    decoder.dec_emb_dim = 128            # Reduced: 256 -> 128
    decoder.dec_depth = 2                # Reduced: 4 -> 2
    decoder.dec_num_heads = 4            # Reduced: 8 -> 4
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2           # MLP layers in output head
    decoder.out_dim = 108                # Same as input features
    decoder.fourier_freq = 10.0          # RFF scale for ky embedding
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_flux_dit_config():
    """Configuration for the Diffusion Transformer (Rectified Flow).

    Input: z_t (B, num_latents, emb_dim) - noisy latent from Target FAE
    Condition: z_c (B, z_c_dim) - TGLF conditioning vector
    Output: velocity (B, num_latents, emb_dim)
    """
    config = ml_collections.ConfigDict()
    config.model_name = "FLUX_DiT"

    config.emb_dim = 128                 # Must match Target FAE latent dim (v2: 128)
    config.num_latents = 32              # Must match Target FAE num_latents (v2: 32)
    config.depth = 6                     # Number of DiT blocks
    config.num_heads = 4                 # Proportional to emb_dim
    config.mlp_ratio = 4
    config.out_dim = 128                 # Must match emb_dim
    config.z_c_dim = 128                 # Must match TGLF FAE z_c_dim (v2: 128)

    return config

