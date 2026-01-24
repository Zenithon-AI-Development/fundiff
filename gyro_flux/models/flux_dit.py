"""Flux Diffusion Transformer (FluxDiT) for generating flux latents.

This module implements a DiT model that learns to generate flux latents z_1
from noise, conditioned on TGLF stability vectors z_c.

Key design: Global modulation via AdaLN-Zero (not input concatenation).
Time and condition embeddings are combined and used to modulate layer statistics.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

import flax.linen as nn

# Import utilities from function_diffusion
from function_diffusion.models.dit import (
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
    modulate,
    DiTBlock,
    FinalLayer,
    MlpBlock,
)


class FluxDiT(nn.Module):
    """Diffusion Transformer for flux latent generation.
    
    Inputs:
        z_t: (B, num_latents, emb_dim) - noisy latent from Target FAE
        t: (B,) - diffusion timestep in [0, 1]
        z_c: (B, z_c_dim) - TGLF conditioning vector
    
    Output:
        velocity: (B, num_latents, emb_dim) - predicted velocity for rectified flow
    
    Architecture:
        1. Project z_t to emb_dim and add positional embeddings
        2. Embed time t and condition z_c separately
        3. Combine time + condition embeddings (element-wise sum)
        4. Pass combined embedding to DiT blocks for AdaLN-Zero modulation
        5. Final layer outputs velocity prediction
    """
    emb_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    z_c_dim: int = 256  # TGLF condition dimension
    num_latents: int = 64  # Sequence length (from Target FAE)
    
    @nn.compact
    def __call__(self, z_t, t, z_c):
        """
        Args:
            z_t: (B, num_latents, emb_dim) noisy latent
            t: (B,) diffusion timestep [0, 1]
            z_c: (B, z_c_dim) TGLF conditioning vector
        
        Returns:
            velocity: (B, num_latents, emb_dim) predicted velocity
        """
        b, l, d = z_t.shape
        
        # 1. Project input and add positional embeddings
        x = nn.Dense(self.emb_dim)(z_t)  # (B, num_latents, emb_dim)
        
        # Add positional embeddings for sequence positions
        pos_emb = self.variable(
            "pos_emb",
            "master_pe",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            self.num_latents,
        )
        x = x + pos_emb.value  # (B, num_latents, emb_dim)
        
        # 2. Embed time and condition separately
        t_emb = TimestepEmbedder(self.emb_dim)(t)  # (B, emb_dim)
        c_emb = nn.Dense(self.emb_dim)(z_c)  # (B, emb_dim)
        
        # 3. Combine time and condition embeddings (element-wise sum)
        # This combined vector drives AdaLN-Zero modulation
        cond = t_emb + c_emb  # (B, emb_dim)
        
        # 4. Pass through DiT blocks with combined conditioning
        for _ in range(self.depth):
            x = DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)(x, cond)
        
        # 5. Final layer with AdaLN-Zero
        x = FinalLayer(self.emb_dim, self.emb_dim)(x, cond)
        
        return x
