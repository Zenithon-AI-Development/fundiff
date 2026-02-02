"""Time-Series Function Autoencoder for CGYRO flux history.

This module implements the Target History FAE (Module B) which:
- Encodes (B, T, 2) flux time-series + (B, T, 1) time coordinates into latent (B, num_latents, emb_dim)
  where T varies from ~150 to ~3000 timesteps
- Decodes latent + continuous time queries into flux values at arbitrary timesteps

Key design choices:
- Index-invariant: Uses explicit physical time coordinates, not array indices
- Time coordinate: τ = (t - 3.0) / 1000.0 (normalized relative to global phase cutoff)
  This preserves physics: growth phase always at consistent τ regardless of file start time
- Fourier embeddings for time: Both encoder and decoder use same Fourier frequency
- Softplus output for non-negative heat flux
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

import flax.linen as nn
from einops import rearrange, repeat

# Reuse components from function_diffusion
from function_diffusion.models.fae import (
    get_1d_sincos_pos_embed,
    PerceiverBlock,
    SelfAttnBlock,
    CrossAttnBlock,
    FourierEmbs,
    Mlp,
    # Skip-Perceiver components
    create_attention_mask,
    MaskedSelfAttnBlock,
    MaskedCrossAttnBlock,
    FiLMConditioning,
    ReadoutPooling,
)


class PatchEmbed1D(nn.Module):
    """1D Patch Embedding for time-series.
    
    Groups consecutive timesteps into patches using 1D convolution.
    (B, T, C) -> (B, T//patch_size, emb_dim)
    """
    patch_size: int = 5
    emb_dim: int = 256
    use_norm: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, T, C) input time-series
        Returns:
            (B, num_patches, emb_dim)
        """
        b, t, c = x.shape
        
        # 1D Conv with kernel_size=patch_size, stride=patch_size
        x = nn.Conv(
            features=self.emb_dim,
            kernel_size=(self.patch_size,),
            strides=(self.patch_size,),
            kernel_init=self.kernel_init,
            name="proj",
        )(x)  # (B, T//patch_size, emb_dim)
        
        if self.use_norm:
            x = nn.LayerNorm(epsilon=1e-5, name="norm")(x)
        
        return x


class TimeSeriesEncoder(nn.Module):
    """Encoder for CGYRO flux time-series with variable length.
    
    Index-invariant encoder that uses explicit physical time coordinates.
    No longer relies on array indices or positional embeddings.
    
    Pipeline:
        1. Pointwise Flux Embedding: (B, T, 2) -> (B, T, emb_dim) via Dense projection
        2. Fourier Time Embedding: (B, T, 1) -> (B, T, emb_dim) via Fourier features
        3. Fuse: element-wise addition of flux and time embeddings
        4. Perceiver Bottleneck: variable T -> fixed num_latents
        5. [Optional] Physics Conditioning: concatenate physics token to latents
        6. Self-Attention Transformer Blocks
    
    Input: 
        x: (B, T, 2) flux time-series, T varies (or fixed if padded)
        t: (B, T, 1) normalized time coordinates τ (already normalized by dataset)
        physics_params: Optional (B, num_physics_params) plasma physics parameters
    Output: (B, num_latents [+1 if physics], emb_dim)
    
    Note on masking: When using padded sequences, padded positions are 0-valued.
    The model learns to ignore these through training. For proper attention masking,
    the PerceiverBlock would need to be extended to accept key masks.
    """
    # Input specs
    in_channels: int = 2
    emb_dim: int = 256
    
    # Perceiver bottleneck
    num_latents: int = 64
    perceiver_depth: int = 2
    
    # Transformer
    transformer_depth: int = 6
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5
    
    # Fourier embedding for time (must match decoder)
    fourier_freq: float = 50.0  # High frequency for turbulent oscillations
    
    # Physics conditioning (NEW)
    use_physics_conditioning: bool = False
    num_physics_params: int = 5  # DLNTDR_1, DLNNDR_1, KY, NU_EE, MASS_1

    @nn.compact
    def __call__(self, x, t, physics_params=None, mask=None):
        """
        Args:
            x: (B, T, C) input flux time-series, T varies or fixed
            t: (B, T, 1) normalized time coordinates τ (from dataset)
            physics_params: Optional (B, num_physics_params) normalized physics parameters
            mask: Optional (B, T) mask where 1 = valid, 0 = padded (currently unused)
        Returns:
            z: (B, num_latents [+1], emb_dim) latent representation
        """
        b, t_seq, c = x.shape
        
        # 1. Pointwise Flux Embedding (preserves temporal resolution)
        x_emb = nn.Dense(self.emb_dim, name="flux_proj")(x)  # (B, T, emb_dim)
        
        # 2. Fourier Time Embedding
        # Ensure t has correct shape: (B, T, 1)
        if t.ndim == 2:
            t = t[..., None]  # (B, T) -> (B, T, 1)
        
        t_emb = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.emb_dim,
        )(t)  # (B, T, emb_dim)
        
        # 3. Fuse flux and time embeddings (element-wise addition)
        x = x_emb + t_emb  # (B, T, emb_dim)
        
        # 4. Perceiver Bottleneck: compress variable T to fixed num_latents
        x = PerceiverBlock(
            emb_dim=self.emb_dim,
            depth=self.perceiver_depth,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
        )(x)  # (B, num_latents, emb_dim)
        
        # 5. Physics Conditioning (NEW): project physics params and concatenate as extra token
        if self.use_physics_conditioning and physics_params is not None:
            # Two-layer MLP to project physics params to emb_dim
            physics_emb = nn.Dense(self.emb_dim, name="physics_proj1")(physics_params)  # (B, emb_dim)
            physics_emb = nn.gelu(physics_emb)
            physics_emb = nn.Dense(self.emb_dim, name="physics_proj2")(physics_emb)      # (B, emb_dim)
            
            # Expand to token format: (B, emb_dim) -> (B, 1, emb_dim)
            physics_token = physics_emb[:, None, :]  # (B, 1, emb_dim)
            
            # Concatenate physics token to dynamics latents
            x = jnp.concatenate([x, physics_token], axis=1)  # (B, num_latents + 1, emb_dim)
        
        # 6. Self-Attention Transformer Blocks
        for _ in range(self.transformer_depth):
            x = SelfAttnBlock(
                num_heads=self.num_heads,
                emb_dim=self.emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(x)
        
        # Final LayerNorm
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        return x


class TimeSeriesEncoderSkipPerceiver(nn.Module):
    """Encoder for CGYRO flux WITHOUT Perceiver bottleneck.

    Skip-Perceiver architecture that performs full self-attention on the
    sequence before compressing to fixed-size representation. This avoids
    information loss from early bottlenecking.

    Key differences from TimeSeriesEncoder:
    - No Perceiver block: full sequence self-attention first
    - Explicit attention masking: padded positions get -inf attention scores
    - FiLM physics conditioning: applied early (before attention), not post-hoc
    - ReadoutPooling: compresses AFTER self-attention via learnable query tokens

    Pipeline:
        1. Pointwise Flux Embedding: (B, T, 2) -> (B, T, emb_dim)
        2. Fourier Time Embedding: (B, T, 1) -> (B, T, emb_dim)
        3. Fuse: element-wise addition
        4. FiLM Physics Conditioning: scale/shift features based on physics params
        5. Masked Self-Attention Transformer: full sequence with explicit masking
        6. ReadoutPooling: aggregate to fixed-size via cross-attention

    Input:
        x: (B, T_pad, 2) flux time-series (padded to max length)
        t: (B, T_pad, 1) normalized time coordinates τ
        physics_params: Optional (B, num_physics_params) plasma physics parameters
        mask: Optional (B, T_pad) validity mask (1=valid, 0=padding)
        lengths: Optional (B,) actual sequence lengths
    Output: (B, num_readout, emb_dim)
    """
    # Input specs
    in_channels: int = 2
    emb_dim: int = 256

    # Readout pooling (replaces Perceiver)
    num_readout: int = 64  # Number of output tokens (like num_latents)
    readout_depth: int = 2  # Cross-attention layers in readout

    # Transformer
    transformer_depth: int = 6
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    # Fourier embedding for time (must match decoder)
    fourier_freq: float = 50.0

    # Physics conditioning
    use_physics_conditioning: bool = False
    num_physics_params: int = 5  # DLNTDR_1, DLNNDR_1, KY, NU_EE, MASS_1
    physics_conditioning_type: str = "film"  # "film", "token", or "none"

    @nn.compact
    def __call__(self, x, t, physics_params=None, mask=None, lengths=None):
        """
        Args:
            x: (B, T_pad, C) input flux time-series (padded)
            t: (B, T_pad, 1) normalized time coordinates τ
            physics_params: Optional (B, num_physics_params) normalized physics parameters
            mask: Optional (B, T_pad) validity mask (True=valid, False=padding)
            lengths: Optional (B,) actual sequence lengths (used to create mask)
        Returns:
            z: (B, num_readout, emb_dim) latent representation
        """
        b, t_seq, c = x.shape

        # Create attention mask from lengths if provided
        if lengths is not None and mask is None:
            # Create 1D mask from lengths: (B, T_pad)
            positions = jnp.arange(t_seq)
            mask = positions[None, :] < lengths[:, None]  # (B, T_pad)

        # Create 2D attention mask for self-attention
        if lengths is not None:
            attn_mask = create_attention_mask(lengths, t_seq)  # (B, 1, T, T)
        else:
            attn_mask = None

        # 1. Pointwise Flux Embedding
        x_emb = nn.Dense(self.emb_dim, name="flux_proj")(x)  # (B, T_pad, emb_dim)

        # 2. Fourier Time Embedding
        if t.ndim == 2:
            t = t[..., None]

        t_emb = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.emb_dim,
        )(t)  # (B, T_pad, emb_dim)

        # 3. Fuse flux and time embeddings
        x = x_emb + t_emb  # (B, T_pad, emb_dim)

        # 4. Physics Conditioning (EARLY - before self-attention)
        if self.use_physics_conditioning and physics_params is not None:
            if self.physics_conditioning_type == "film":
                # FiLM: modulate ALL tokens with physics-based scale/shift
                x = FiLMConditioning(emb_dim=self.emb_dim)(x, physics_params)
            elif self.physics_conditioning_type == "token":
                # Token: add physics as extra token (prepend so mask is simpler)
                physics_emb = nn.Dense(self.emb_dim, name="physics_proj1")(physics_params)
                physics_emb = nn.gelu(physics_emb)
                physics_emb = nn.Dense(self.emb_dim, name="physics_proj2")(physics_emb)
                physics_token = physics_emb[:, None, :]  # (B, 1, emb_dim)
                x = jnp.concatenate([physics_token, x], axis=1)  # (B, T_pad+1, emb_dim)

                # Update mask/attn_mask for the extra token
                if mask is not None:
                    physics_mask = jnp.ones((b, 1), dtype=mask.dtype)
                    mask = jnp.concatenate([physics_mask, mask], axis=1)
                if lengths is not None:
                    # Recompute attn_mask with +1 for physics token
                    new_lengths = lengths + 1
                    attn_mask = create_attention_mask(new_lengths, t_seq + 1)

        # 5. Masked Self-Attention Transformer (full sequence)
        for _ in range(self.transformer_depth):
            x = MaskedSelfAttnBlock(
                num_heads=self.num_heads,
                emb_dim=self.emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(x, mask=attn_mask)

        # 6. ReadoutPooling: aggregate to fixed-size representation
        # Uses cross-attention with learnable readout tokens as queries
        z = ReadoutPooling(
            num_readout=self.num_readout,
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            depth=self.readout_depth,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
        )(x, mask=mask)  # (B, num_readout, emb_dim)

        return z


class ContinuousTimeDecoder(nn.Module):
    """Continuous-time decoder for flux prediction.
    
    Index-invariant decoder that accepts explicit continuous time queries.
    No longer converts step indices; expects normalized τ directly.
    
    Pipeline:
        1. Fourier embed τ: (B, N_q, 1) -> (B, N_q, dec_emb_dim)
        2. Cross-Attention Stack: queries attend to latent
        3. MLP Head -> out_channels
        4. Softplus Activation (non-negativity)
    
    Input: 
        z: (B, num_latents, latent_dim) latent representation
        t_query: (B, N_q, 1) or (B, N_q) normalized time coordinates τ (from dataset)
    Output:
        flux: (B, N_q, out_channels) predicted flux values
    """
    # Dimensions
    latent_dim: int = 256
    dec_emb_dim: int = 256
    out_channels: int = 2
    
    # Architecture
    dec_depth: int = 6
    dec_num_heads: int = 8
    mlp_ratio: int = 2
    num_mlp_layers: int = 2
    layer_norm_eps: float = 1e-5
    
    # Fourier embedding (must match encoder)
    fourier_freq: float = 50.0  # High frequency for turbulent oscillations
    
    # Physics constraint
    use_softplus: bool = True

    @nn.compact
    def __call__(self, z, t_query):
        """
        Args:
            z: (B, num_latents, latent_dim) latent representation
            t_query: (B, N_q, 1) or (B, N_q) normalized time coordinates τ (already normalized)
        Returns:
            flux: (B, N_q, out_channels) predicted flux values
        """
        b, n_lat, d = z.shape
        
        # Ensure t_query has correct shape: (B, N_q, 1)
        if t_query.ndim == 1:
            # (N_q,) -> (B, N_q, 1)
            t_query = repeat(t_query, 'n -> b n', b=b)[..., None]
        elif t_query.ndim == 2:
            # (B, N_q) -> (B, N_q, 1)
            t_query = t_query[..., None]
        
        # 1. Fourier embed τ (same configuration as encoder)
        q = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.dec_emb_dim,
        )(t_query)  # (B, N_q, dec_emb_dim)
        
        # 2. Project latent to dec_emb_dim if needed
        kv = z
        if self.latent_dim != self.dec_emb_dim:
            kv = nn.Dense(self.dec_emb_dim, name="latent_proj")(z)
        
        # 3. Cross-Attention Stack
        for i in range(self.dec_depth):
            q = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(q, kv)
        
        # Final LayerNorm
        q = nn.LayerNorm(epsilon=self.layer_norm_eps, name="dec_norm")(q)
        
        # 4. MLP Output Head
        flux = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_channels,
            layer_norm_eps=self.layer_norm_eps,
        )(q)  # (B, N_q, out_channels)
        
        # 5. Softplus for non-negativity (heat flux cannot be negative)
        if self.use_softplus:
            flux = nn.softplus(flux)
        
        return flux


class TimeFAE(nn.Module):
    """Complete Time-Series Function Autoencoder.
    
    Wrapper combining encoder and decoder for convenience.
    """
    encoder: TimeSeriesEncoder
    decoder: ContinuousTimeDecoder

    @nn.compact
    def __call__(self, x, t, t_query):
        """
        Args:
            x: (B, T, C) input flux time-series (T varies)
            t: (B, T, 1) normalized time coordinates for input sequence
            t_query: (B, N_q, 1) or (B, N_q) normalized time coordinates to reconstruct
        Returns:
            flux: (B, N_q, out_channels) reconstructed flux at query times
        """
        z = self.encoder(x, t)
        flux = self.decoder(z, t_query)
        return flux

