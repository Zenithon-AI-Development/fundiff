from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def _to_channels_first(u: jnp.ndarray, channels: int = 4) -> jnp.ndarray:
    """Ensure tensor is channels-first (B, C, H, W) for loss computations."""
    if u.ndim != 4:
        raise ValueError(f"Expected rank-4 tensor, got shape={u.shape}")
    # Either (B, C, H, W) or (B, H, W, C)
    if u.shape[1] == channels:
        return u
    if u.shape[-1] == channels:
        return jnp.transpose(u, (0, 3, 1, 2))
    raise ValueError(f"Expected channels={channels} on axis 1 or -1, got shape={u.shape}")


def reconstruction_loss(
    u_hat: jnp.ndarray,
    u_gt: jnp.ndarray,
    kind: str = "l1",
    huber_delta: float = 1.0,
) -> jnp.ndarray:
    """Per-element reconstruction loss averaged over all elements."""
    diff = u_hat - u_gt
    kind = kind.lower()
    if kind == "l1":
        return jnp.mean(jnp.abs(diff))
    if kind in ("l2", "mse"):
        return jnp.mean(diff**2)
    if kind == "huber":
        abs_diff = jnp.abs(diff)
        quadratic = jnp.minimum(abs_diff, huber_delta)
        linear = abs_diff - quadratic
        return jnp.mean(0.5 * quadratic**2 + huber_delta * linear)
    raise ValueError(f"Unknown reconstruction loss kind: {kind}")


def _radial_bin_indices(
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    num_bins: int,
) -> jnp.ndarray:
    """Compute integer radial bins from kx, ky (shared across batch)."""
    r = jnp.sqrt(kx**2 + ky**2)
    # "integer bins of r" -> floor(r)
    bins = jnp.floor(r).astype(jnp.int32)
    return jnp.clip(bins, 0, num_bins - 1)


def _radial_spectrum_sum(
    energy: jnp.ndarray,
    bin_idx_hw: jnp.ndarray,
    num_bins: int,
) -> jnp.ndarray:
    """Sum energy into radial bins.

    Args:
        energy: (B, H, W)
        bin_idx_hw: (H, W) integer bin ids
    Returns:
        spec: (B, num_bins)
    """
    if energy.ndim != 3:
        raise ValueError(f"Expected energy (B,H,W), got shape={energy.shape}")
    if bin_idx_hw.ndim != 2:
        raise ValueError(f"Expected bin_idx_hw (H,W), got shape={bin_idx_hw.shape}")

    seg_ids = bin_idx_hw.reshape(-1)  # (H*W,)
    e_flat = energy.reshape(energy.shape[0], -1)  # (B, H*W)

    def _seg_sum(e1d):
        return jax.lax.segment_sum(e1d, seg_ids, num_segments=num_bins)

    return jax.vmap(_seg_sum)(e_flat)


def spectral_density_loss(
    u_hat: jnp.ndarray,
    u_gt: jnp.ndarray,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    num_bins: int = 128,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Log-spectrum MSE loss for phi and A_parallel magnitudes.

    u_* expected channels-first (B, 4, H, W).
    kx, ky expected shape (H, W) (coordinate grid).
    """
    u_hat = _to_channels_first(u_hat, channels=4)
    u_gt = _to_channels_first(u_gt, channels=4)

    # energies: (B, H, W)
    e_phi_hat = u_hat[:, 0] ** 2 + u_hat[:, 1] ** 2
    e_phi_gt = u_gt[:, 0] ** 2 + u_gt[:, 1] ** 2
    e_a_hat = u_hat[:, 2] ** 2 + u_hat[:, 3] ** 2
    e_a_gt = u_gt[:, 2] ** 2 + u_gt[:, 3] ** 2

    bin_idx = _radial_bin_indices(kx, ky, num_bins=num_bins)  # (H, W)
    s_phi_hat = _radial_spectrum_sum(e_phi_hat, bin_idx, num_bins)  # (B, K)
    s_phi_gt = _radial_spectrum_sum(e_phi_gt, bin_idx, num_bins)
    s_a_hat = _radial_spectrum_sum(e_a_hat, bin_idx, num_bins)
    s_a_gt = _radial_spectrum_sum(e_a_gt, bin_idx, num_bins)

    log_phi_hat = jnp.log(s_phi_hat + eps)
    log_phi_gt = jnp.log(s_phi_gt + eps)
    log_a_hat = jnp.log(s_a_hat + eps)
    log_a_gt = jnp.log(s_a_gt + eps)

    return jnp.mean((log_phi_hat - log_phi_gt) ** 2) + jnp.mean((log_a_hat - log_a_gt) ** 2)


def cross_spectrum_loss(u_hat: jnp.ndarray, u_gt: jnp.ndarray) -> jnp.ndarray:
    """Phase-sensitive cross-spectrum loss:
      C = conj(phi) * A_parallel, compare complex values in the plane via MSE on Re/Im.
    """
    u_hat = _to_channels_first(u_hat, channels=4)
    u_gt = _to_channels_first(u_gt, channels=4)

    phi_hat = u_hat[:, 0] + 1j * u_hat[:, 1]
    a_hat = u_hat[:, 2] + 1j * u_hat[:, 3]
    phi_gt = u_gt[:, 0] + 1j * u_gt[:, 1]
    a_gt = u_gt[:, 2] + 1j * u_gt[:, 3]

    c_hat = jnp.conj(phi_hat) * a_hat
    c_gt = jnp.conj(phi_gt) * a_gt

    return jnp.mean((jnp.real(c_hat) - jnp.real(c_gt)) ** 2) + jnp.mean(
        (jnp.imag(c_hat) - jnp.imag(c_gt)) ** 2
    )


def compute_gyro_ae_losses(
    u_hat: jnp.ndarray,
    u_gt: jnp.ndarray,
    coords_grid: Optional[jnp.ndarray] = None,
    *,
    rec_kind: str = "l1",
    rec_huber_delta: float = 1.0,
    w_rec: float = 1.0,
    w_spec: float = 0.0,
    w_cross: float = 0.0,
    spec_num_bins: int = 128,
    spec_eps: float = 1e-8,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute total loss + components for gyro autoencoder.

    Notes:
      - `u_hat` and `u_gt` can be channels-first (B,4,H,W) or channels-last (B,H,W,4).
      - `coords_grid` (H,W,2) or (B,H,W,2) is required if w_spec > 0 (to get kx,ky bins).
    """
    u_hat_cf = _to_channels_first(u_hat, channels=4)
    u_gt_cf = _to_channels_first(u_gt, channels=4)

    l_rec = reconstruction_loss(u_hat_cf, u_gt_cf, kind=rec_kind, huber_delta=rec_huber_delta)

    l_spec = jnp.asarray(0.0, dtype=u_hat_cf.dtype)
    if w_spec != 0.0:
        if coords_grid is None:
            raise ValueError("coords_grid is required for spectral_density_loss when w_spec != 0.")
        if coords_grid.ndim == 4:
            coords_hw = coords_grid[0]
        else:
            coords_hw = coords_grid
        kx = coords_hw[..., 0]
        ky = coords_hw[..., 1]
        l_spec = spectral_density_loss(
            u_hat_cf, u_gt_cf, kx=kx, ky=ky, num_bins=spec_num_bins, eps=spec_eps
        )

    l_cross = jnp.asarray(0.0, dtype=u_hat_cf.dtype)
    if w_cross != 0.0:
        l_cross = cross_spectrum_loss(u_hat_cf, u_gt_cf)

    total = w_rec * l_rec + w_spec * l_spec + w_cross * l_cross
    metrics = {
        "loss_total": total,
        "loss_rec": l_rec,
        "loss_spec": l_spec,
        "loss_cross": l_cross,
    }
    return total, metrics


