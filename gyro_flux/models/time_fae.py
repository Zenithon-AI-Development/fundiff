"""Time-Series Function Autoencoder for CGYRO flux history.

This module implements the Target History FAE (Module B) which:
- Encodes (B, T, 2) flux time-series into latent (B, num_latents, emb_dim)
  where T varies from ~150 to ~3000 timesteps
- Decodes latent + time queries into flux values at arbitrary timesteps

Key design choices:
- Variable-length input handled via interpolated positional embeddings
- Time coordinate: τ = step_index / time_scale (not normalized to [0,1])
  This preserves physics: growth phase always at τ ≈ 0-0.025
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
    
    Handles variable-length input (T = 150 to 3000) via interpolated
    positional embeddings, similar to the TGLF encoder approach.
    
    Pipeline:
        1. 1D Patch Embedding: (B, T, 2) -> (B, T//patch_size, emb_dim)
        2. Interpolated Positional Embeddings (from master table)
        3. Perceiver Bottleneck: variable patches -> fixed num_latents
        4. Self-Attention Transformer Blocks
    
    Input: (B, T, in_channels) where T varies (or fixed if padded)
    Output: (B, num_latents, emb_dim)
    
    Note on masking: When using padded sequences, padded positions are 0-valued.
    The model learns to ignore these through training. For proper attention masking,
    the PerceiverBlock would need to be extended to accept key masks.
    """
    # Input specs
    in_channels: int = 2
    
    # Patch embedding
    patch_size: int = 5
    emb_dim: int = 256
    
    # Positional embedding (for interpolation)
    max_patches: int = 600  # Max T=3000 / patch_size=5 = 600 patches
    
    # Perceiver bottleneck
    num_latents: int = 64
    perceiver_depth: int = 2
    
    # Transformer
    transformer_depth: int = 6
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5
    
    # Padding mode
    use_fixed_pe: bool = False  # If True, don't interpolate PE (for padded mode)

    @nn.compact
    def __call__(self, x, mask=None):
        """
        Args:
            x: (B, T, C) input time-series, T varies or fixed
            mask: Optional (B, T) mask where 1 = valid, 0 = padded (currently unused)
        Returns:
            z: (B, num_latents, emb_dim) latent representation
        """
        b, t, c = x.shape
        
        # 1. Patch Embedding
        x = PatchEmbed1D(
            patch_size=self.patch_size,
            emb_dim=self.emb_dim,
        )(x)  # (B, num_patches, emb_dim)
        
        num_patches = x.shape[1]
        
        # 2. Positional Embeddings
        if self.use_fixed_pe:
            # Fixed PE for padded mode - no interpolation needed
            # Use first num_patches of master PE
            master_pe = self.variable(
                "pos_emb",
                "master_pe",
                get_1d_sincos_pos_embed,
                self.emb_dim,
                self.max_patches,
            )  # (1, max_patches, emb_dim)
            pe = master_pe.value[:, :num_patches, :]  # (1, num_patches, emb_dim)
        else:
            # Interpolated PE for variable-length mode
            master_pe = self.variable(
                "pos_emb",
                "master_pe",
                get_1d_sincos_pos_embed,
                self.emb_dim,
                self.max_patches,
            )  # (1, max_patches, emb_dim)
            
            pe = jax.image.resize(
                master_pe.value,
                shape=(1, num_patches, self.emb_dim),
                method='bilinear',
            )  # (1, num_patches, emb_dim)
        
        x = x + pe  # (B, num_patches, emb_dim)
        
        # 3. Perceiver Bottleneck: compress variable patches to fixed num_latents
        x = PerceiverBlock(
            emb_dim=self.emb_dim,
            depth=self.perceiver_depth,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
        )(x)  # (B, num_latents, emb_dim)
        
        # 4. Self-Attention Transformer Blocks
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


class ContinuousTimeDecoder(nn.Module):
    """Continuous-time decoder for flux prediction.
    
    Given a latent representation and timestep queries, predicts flux values.
    Uses τ = step_index / time_scale as the time coordinate to preserve
    physics (growth phase at consistent τ regardless of total length).
    
    Pipeline:
        1. Convert step indices to τ: τ = step_index / time_scale
        2. Fourier embed τ: (B, N_q, 1) -> (B, N_q, dec_emb_dim)
        3. Cross-Attention Stack: queries attend to latent
        4. MLP Head -> out_channels
        5. Softplus Activation (non-negativity)
    
    Input: 
        z: (B, num_latents, latent_dim) latent representation
        step_query: (B, N_q) or (N_q,) integer step indices [0, 1, 2, ...]
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
    
    # Time coordinate normalization
    time_scale: float = 1000.0  # τ = step_index / time_scale
    
    # Fourier embedding
    fourier_freq: float = 50.0  # High frequency for turbulent oscillations
    
    # Physics constraint
    use_softplus: bool = True

    @nn.compact
    def __call__(self, z, step_query):
        """
        Args:
            z: (B, num_latents, latent_dim) latent representation
            step_query: (B, N_q) or (N_q,) integer step indices
        Returns:
            flux: (B, N_q, out_channels) predicted flux values
        """
        b, n_lat, d = z.shape
        
        # Handle step_query shape
        if step_query.ndim == 1:
            # (N_q,) -> (B, N_q)
            step_query = repeat(step_query, 'n -> b n', b=b)
        
        # Convert to normalized time coordinate
        # τ = step_index / time_scale (e.g., step 25 -> τ = 0.025)
        tau = step_query.astype(jnp.float32) / self.time_scale  # (B, N_q)
        tau = tau[..., None]  # (B, N_q, 1)
        
        # 1. Fourier embed τ
        q = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.dec_emb_dim,
        )(tau)  # (B, N_q, dec_emb_dim)
        
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
    def __call__(self, x, step_query):
        """
        Args:
            x: (B, T, C) input time-series (T varies)
            step_query: (B, N_q) or (N_q,) step indices to reconstruct
        Returns:
            flux: (B, N_q, out_channels) reconstructed flux at query times
        """
        z = self.encoder(x)
        flux = self.decoder(z, step_query)
        return flux

