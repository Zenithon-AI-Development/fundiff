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
    
    Variable-length input: (B, T, 2) where T ranges ~150 to ~3000 timesteps
    Time coordinate: τ = step_index / time_scale (preserves physics)
    Latent: (B, num_latents, emb_dim)
    Output: (B, N_queries, 2) at arbitrary timesteps
    """
    config = ml_collections.ConfigDict()
    config.model_name = "TARGET_FAE"

    # Encoder: (B, T, 2) -> (B, num_latents, emb_dim), T varies
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.in_channels = 2              # Q_i, Q_e
    encoder.patch_size = 5               # Group 5 timesteps per patch
    encoder.emb_dim = 256                # Embedding dimension
    encoder.max_patches = 600            # Max T=3000 / patch_size=5 for PE interpolation
    encoder.num_latents = 64             # Latent sequence length after Perceiver
    encoder.perceiver_depth = 2          # Cross-attn layers in Perceiver bottleneck
    encoder.transformer_depth = 6        # Self-attn layers after bottleneck
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5
    encoder.use_fixed_pe = False         # True for padded mode (no PE interpolation)

    # Decoder: (B, num_latents, emb_dim) + step_query -> (B, N_queries, 2)
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.latent_dim = 256             # Must match encoder.emb_dim
    decoder.dec_emb_dim = 256            # Decoder embedding dimension
    decoder.dec_depth = 6                # Cross-attn layers (deep for high-freq wiggles)
    decoder.dec_num_heads = 8
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2           # MLP layers in output head
    decoder.out_channels = 2             # Q_i, Q_e
    decoder.time_scale = 1000.0          # τ = step_index / time_scale
    decoder.fourier_freq = 50.0          # RFF scale (high for turbulent oscillations)
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
    """
    config = ml_collections.ConfigDict()
    config.model_name = "TGLF_FAE"

    # Encoder: (B, 21, 108) -> z_c (B, z_c_dim)
    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.num_ky_modes = 21            # Fixed number of ky modes
    encoder.input_dim = 108              # Fixed features per mode
    encoder.emb_dim = 256                # Embedding dimension
    encoder.num_latents = 16             # Latent sequence length after Perceiver
    encoder.perceiver_depth = 2          # Cross-attn layers in Perceiver bottleneck
    encoder.transformer_depth = 4        # Self-attn layers after bottleneck
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5
    # Aggregation to z_c
    encoder.z_c_aggregation = "mean"     # "mean" or "flatten"
    encoder.z_c_dim = 256                # Output z_c dimension (same as emb_dim when no projection)
    encoder.use_z_c_projection = False   # No projection - mean pooling gives (B, 256) directly

    # Decoder: z_c (B, z_c_dim) + ky_query -> (B, N_queries, 108)
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.z_c_dim = 256                # Must match encoder.z_c_dim
    decoder.dec_emb_dim = 256            # Decoder embedding dimension
    decoder.dec_depth = 4                # MLP depth (no cross-attn, just conditioned MLP)
    decoder.dec_num_heads = 8            # Unused now, kept for compatibility
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

    config.emb_dim = 256                 # Must match Target FAE latent dim
    config.num_latents = 64              # Must match Target FAE num_latents
    config.depth = 8                     # Number of DiT blocks
    config.num_heads = 8
    config.mlp_ratio = 4
    config.out_dim = 256                 # Must match emb_dim
    config.z_c_dim = 256                 # Must match TGLF FAE z_c_dim

    return config

