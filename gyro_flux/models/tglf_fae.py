"""TGLF Function Autoencoder for conditioning.

This module implements the TGLF FAE (Module A) which:
- Encodes (B, 21, 108) TGLF spectra into z_c (B, z_c_dim)
- Decodes z_c + ky queries into reconstructed features

The encoder directly outputs z_c, the conditioning vector for the DiT,
so the FAE is trained to optimize z_c representation quality.

Fixed input shape: 21 ky modes × 108 features per mode.
Features: 80 ql_weights + 4 gamma + 4 freq + 20 flux predictions = 108
"""

from typing import Callable, Literal

import jax.numpy as jnp
from jax.nn.initializers import normal

import flax.linen as nn
from einops import repeat, rearrange

# Reuse components from function_diffusion
from function_diffusion.models.fae import (
    get_1d_sincos_pos_embed,
    PerceiverBlock,
    SelfAttnBlock,
    FourierEmbs,
    Mlp,
)


class TGLFEncoder(nn.Module):
    """Encoder for TGLF linear spectra.
    
    Fixed input: (B, 21, 108) → z_c (B, z_c_dim)
    
    Outputs z_c directly so the FAE is trained to optimize the conditioning vector.
    
    Pipeline:
        1. Linear Projection: (B, 21, 108) → (B, 21, emb_dim)
        2. Add Fixed Positional Embeddings (21 positions)
        3. Perceiver Bottleneck: (B, 21, emb_dim) → (B, num_latents, emb_dim)
        4. Self-Attention Transformer Blocks
        5. Aggregation: (B, num_latents, emb_dim) → z_c (B, z_c_dim)
    """
    # Input specs (fixed)
    num_ky_modes: int = 21
    input_dim: int = 108
    
    # Embedding
    emb_dim: int = 256
    
    # Perceiver bottleneck
    num_latents: int = 16
    perceiver_depth: int = 2
    
    # Transformer
    transformer_depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5
    
    # z_c aggregation
    z_c_aggregation: Literal["mean", "flatten"] = "mean"
    z_c_dim: int = 256
    use_z_c_projection: bool = True

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (B, 21, 108) TGLF spectra
        Returns:
            z_c: (B, z_c_dim) conditioning vector for DiT
        """
        b, n, f = x.shape
        assert n == self.num_ky_modes, f"Expected {self.num_ky_modes} ky modes, got {n}"
        assert f == self.input_dim, f"Expected {self.input_dim} features, got {f}"
        
        # 1. Linear Projection
        x = nn.Dense(
            self.emb_dim,
            kernel_init=nn.initializers.xavier_uniform(),
            name="input_proj",
        )(x)  # (B, 21, emb_dim)
        
        # 2. Add Fixed Positional Embeddings
        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            self.num_ky_modes,
        )
        x = x + pos_emb.value  # (B, 21, emb_dim)
        
        # 3. Perceiver Bottleneck: compress 21 → num_latents
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
        
        # Final LayerNorm before aggregation
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        
        # 5. Aggregate to z_c
        if self.z_c_aggregation == "mean":
            # Mean pooling: (B, num_latents, emb_dim) → (B, emb_dim)
            z_c = jnp.mean(x, axis=1)
        elif self.z_c_aggregation == "flatten":
            # Flatten: (B, num_latents, emb_dim) → (B, num_latents * emb_dim)
            z_c = rearrange(x, 'b n d -> b (n d)')
        else:
            raise ValueError(f"Unknown z_c_aggregation: {self.z_c_aggregation}")
        
        # Optional learned projection to z_c_dim
        if self.use_z_c_projection:
            z_c = nn.Dense(
                self.z_c_dim,
                kernel_init=nn.initializers.xavier_uniform(),
                name="z_c_proj",
            )(z_c)
        
        return z_c  # (B, z_c_dim)


class TGLFDecoder(nn.Module):
    """Decoder for TGLF spectra reconstruction from z_c.
    
    Given the conditioning vector z_c and ky query indices, reconstructs features.
    Since z_c is 1D (B, z_c_dim), we use a conditioned MLP approach:
    concatenate z_c with ky embeddings and pass through deep MLP.
    
    Pipeline:
        1. Fourier embed ky queries: (B, N_q, 1) → (B, N_q, dec_emb_dim)
        2. Expand z_c: (B, z_c_dim) → (B, N_q, z_c_dim)
        3. Concatenate: (B, N_q, z_c_dim + dec_emb_dim)
        4. Deep MLP Stack → 108 output features
    
    Input: 
        z_c: (B, z_c_dim) conditioning vector
        ky_query: (B, N_q, 1) or (N_q, 1) ky indices in [0, 20]
    Output:
        features: (B, N_q, out_dim) reconstructed features
    """
    # Dimensions
    z_c_dim: int = 256
    dec_emb_dim: int = 256
    out_dim: int = 108
    
    # Architecture
    dec_depth: int = 4          # Number of hidden layers in decoder MLP
    dec_num_heads: int = 8      # Unused, kept for config compatibility
    mlp_ratio: int = 2
    num_mlp_layers: int = 2     # Final output MLP layers
    layer_norm_eps: float = 1e-5
    
    # Fourier embedding
    fourier_freq: float = 10.0

    @nn.compact
    def __call__(self, z_c, ky_query):
        """
        Args:
            z_c: (B, z_c_dim) conditioning vector
            ky_query: (B, N_q, 1) or (B, N_q) or (N_q,) ky indices (integers 0-20)
        Returns:
            features: (B, N_q, out_dim) reconstructed features
        """
        b = z_c.shape[0]
        
        # Handle ky_query shape
        if ky_query.ndim == 1:
            # (N_q,) → (B, N_q, 1)
            ky_query = repeat(ky_query, 'n -> b n 1', b=b)
        elif ky_query.ndim == 2:
            if ky_query.shape[0] == b:
                # (B, N_q) → (B, N_q, 1)
                ky_query = ky_query[..., None]
            else:
                # (N_q, 1) → (B, N_q, 1)
                ky_query = repeat(ky_query, 'n d -> b n d', b=b)
        
        n_q = ky_query.shape[1]
        
        # Normalize ky indices to [0, 1] for Fourier embedding
        # ky_query values are integers 0-20, normalize to 0-1
        ky_normalized = ky_query.astype(jnp.float32) / 20.0
        
        # 1. Fourier embed ky queries
        ky_emb = FourierEmbs(
            embed_scale=self.fourier_freq,
            embed_dim=self.dec_emb_dim,
        )(ky_normalized)  # (B, N_q, dec_emb_dim)
        
        # 2. Expand z_c to match query dimension
        z_c_expanded = repeat(z_c, 'b d -> b n d', n=n_q)  # (B, N_q, z_c_dim)
        
        # 3. Concatenate z_c with ky embeddings
        x = jnp.concatenate([z_c_expanded, ky_emb], axis=-1)  # (B, N_q, z_c_dim + dec_emb_dim)
        
        # 4. Deep MLP Stack (replaces cross-attention)
        hidden_dim = self.dec_emb_dim * self.mlp_ratio
        for i in range(self.dec_depth):
            x = nn.Dense(hidden_dim, name=f"dec_mlp_{i}")(x)
            x = nn.LayerNorm(epsilon=self.layer_norm_eps, name=f"dec_ln_{i}")(x)
            x = nn.gelu(x)
        
        # 5. Output MLP Head
        features = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )(x)  # (B, N_q, out_dim)
        
        # No activation - TGLF features can be positive or negative
        return features


class TGLFFAE(nn.Module):
    """Complete TGLF Function Autoencoder.
    
    Encodes TGLF spectra to z_c and reconstructs from z_c.
    The FAE is trained to directly optimize z_c quality.
    """
    encoder: TGLFEncoder
    decoder: TGLFDecoder

    @nn.compact
    def __call__(self, x, ky_query):
        """
        Args:
            x: (B, 21, 108) input TGLF spectra
            ky_query: (B, N_q, 1) or (N_q,) ky indices to reconstruct
        Returns:
            features: (B, N_q, 108) reconstructed features at query ky positions
        """
        z_c = self.encoder(x)  # (B, z_c_dim)
        features = self.decoder(z_c, ky_query)  # (B, N_q, 108)
        return features
    
    def encode(self, x):
        """Get conditioning vector z_c for DiT.
        
        Args:
            x: (B, 21, 108) input TGLF spectra
        Returns:
            z_c: (B, z_c_dim) conditioning vector
        """
        return self.encoder(x)

