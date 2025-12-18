from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from function_diffusion.models.fno import FNO2d
from function_diffusion.models.fae import PerceiverBlock, CrossAttnBlock, FourierEmbs, Mlp


class GyroEncoderFNO(nn.Module):
    """Gyrokinetic encoder: (B, 4, Nx, Ny) -> (B, num_latents, latent_dim).

    Pipeline:
      - channel lifting + spectral mixing (FNO)
      - flatten spatial grid into a sequence
      - Perceiver-style latent queries (cross-attention bottleneck)
    """

    # Input/output sizes
    in_channels: int = 4
    latent_dim: int = 128
    num_latents: int = 64

    # FNO backbone
    fno_emb_dim: int = 128  # Hidden_Dim after channel lifting
    fno_depth: int = 4
    modes1: int = 12
    modes2: int = 12
    fno_padding: int = 0
    activation: Callable = nn.gelu

    # Latent bottleneck (Perceiver)
    perceiver_depth: int = 2
    num_heads: int = 8
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Either channels-first (B, C, H, W) or channels-last (B, H, W, C).
        Returns:
            z: (B, num_latents, latent_dim)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected rank-4 input, got shape={x.shape}")

        # Normalize layout to channels-last for FNO2d: (B, H, W, C)
        if x.shape[-1] == self.in_channels:
            x_cl = x
        elif x.shape[1] == self.in_channels:
            x_cl = rearrange(x, "b c h w -> b h w c")
        else:
            raise ValueError(
                f"Input channel dimension mismatch. "
                f"Expected in_channels={self.in_channels} in axis 1 or -1, got shape={x.shape}."
            )

        # FNO backbone (includes channel lifting via Dense -> fno_emb_dim).
        h = FNO2d(
            modes1=self.modes1,
            modes2=self.modes2,
            emb_dim=self.fno_emb_dim,
            out_dim=self.fno_emb_dim,
            depth=self.fno_depth,
            activation=self.activation,
            padding=self.fno_padding,
            model_name="gyro_encoder_fno",
        )(x_cl)  # (B, H, W, fno_emb_dim)

        # Flatten to tokens: (B, H*W, fno_emb_dim)
        h = rearrange(h, "b h w d -> b (h w) d")

        # If desired, map fno_emb_dim -> latent_dim before the Perceiver bottleneck.
        if self.fno_emb_dim != self.latent_dim:
            h = nn.Dense(self.latent_dim, name="pre_latent_proj")(h)

        # Latent queries (cross-attn bottleneck): (B, num_latents, latent_dim)
        z = PerceiverBlock(
            emb_dim=self.latent_dim,
            depth=self.perceiver_depth,
            num_heads=self.num_heads,
            num_latents=self.num_latents,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
        )(h)

        return z


class GyroFAE(nn.Module):
    """Gyrokinetic FAE wrapper.

    Encoder + coordinate-conditioned decoder.
    """

    encoder: GyroEncoderFNO
    decoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, *args, **kwargs):
        z = self.encoder(x)
        if self.decoder is None:
            return z
        return self.decoder(z, *args, **kwargs)


class GyroDecoderCrossAttn(nn.Module):
    """Gyrokinetic decoder: (z, coords) -> U(coords) with conjugate-symmetry enforcement.

    Inputs:
      - z: (B, Num_Latents, Latent_Dim)
      - coords: (B, Num_Queries, 2) where last dim is (kx, ky)

    Output:
      - u: (B, Num_Queries, out_dim) where out_dim=4 by default:
          [Re Phi, Im Phi, Re A_par, Im A_par]
    """

    # Dimensions
    latent_dim: int = 128
    dec_emb_dim: int = 128
    out_dim: int = 4

    # Coordinate embedding
    fourier_freq: float = 10.0

    # Cross-attention stack
    dec_depth: int = 4
    dec_num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    # Output head
    num_mlp_layers: int = 2

    # Physics constraint (conjugate symmetry in k-space)
    enforce_conjugate_symmetry: bool = True
    zero_tol: float = 0.0  # tolerance used for ky==0 comparisons when splitting half-planes

    def _canonicalize_coords(self, coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Map coords to a canonical 'positive half-plane' and return a flip mask.

        For real-valued fields in real-space, Fourier coefficients satisfy:
          U(-kx, -ky) = conj(U(kx, ky)).

        We define the canonical half-plane as:
          ky > 0  OR  (ky == 0 AND kx >= 0)
        Points outside this half-plane are mapped via (kx, ky) -> (-kx, -ky),
        and the output is conjugated afterward.
        """
        kx = coords[..., 0]
        ky = coords[..., 1]
        ky_zero = jnp.abs(ky) <= self.zero_tol
        is_negative_half = (ky < 0.0) | (ky_zero & (kx < 0.0))
        coords_pos = jnp.where(is_negative_half[..., None], -coords, coords)
        return coords_pos, is_negative_half

    def _apply_conjugate(self, u: jnp.ndarray, flip: jnp.ndarray) -> jnp.ndarray:
        """Apply complex conjugation to the (Re, Im) pairs when flip=True.

        Channel convention: [Re Phi, Im Phi, Re A_par, Im A_par].
        Conjugation flips the sign of imaginary components.
        """
        if self.out_dim != 4:
            raise ValueError(
                f"Conjugate symmetry helper assumes out_dim=4 (Re/Im pairs). Got out_dim={self.out_dim}."
            )
        sign = jnp.asarray([1.0, -1.0, 1.0, -1.0], dtype=u.dtype)  # (4,)
        u_conj = u * sign
        return jnp.where(flip[..., None], u_conj, u)

    @nn.compact
    def __call__(self, z: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape (B, L, D), got shape={z.shape}")
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(f"Expected coords with shape (B, Nq, 2), got shape={coords.shape}")

        if self.enforce_conjugate_symmetry:
            coords_used, flip = self._canonicalize_coords(coords)
        else:
            coords_used, flip = coords, jnp.zeros(coords.shape[:-1], dtype=bool)

        # 1) Coordinate embedding: (B, Nq, 2) -> (B, Nq, dec_emb_dim)
        q = FourierEmbs(embed_scale=self.fourier_freq, embed_dim=self.dec_emb_dim)(coords_used)

        # 2) Project latent tokens to decoder embedding dim: (B, L, latent_dim) -> (B, L, dec_emb_dim)
        kv = z
        if z.shape[-1] != self.dec_emb_dim:
            kv = nn.Dense(self.dec_emb_dim, name="latent_proj")(z)

        # 3) Cross-attention stack: coords query the latent tokens
        for _ in range(self.dec_depth):
            q = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
            )(q, kv)

        q = nn.LayerNorm(epsilon=self.layer_norm_eps, name="dec_norm")(q)

        # 4) Projection head: (B, Nq, dec_emb_dim) -> (B, Nq, out_dim)
        u = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
            layer_norm_eps=self.layer_norm_eps,
        )(q)

        # 5) Enforce conjugate symmetry (if requested)
        if self.enforce_conjugate_symmetry:
            u = self._apply_conjugate(u, flip)

        return u


