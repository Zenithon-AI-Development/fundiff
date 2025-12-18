from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax import lax

from gyro.autoencoder.losses import compute_gyro_ae_losses


def decode_full_grid(
    decoder,
    decoder_params,
    z: jnp.ndarray,
    coords_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Decode a full (H,W) coordinate grid into a (B,4,H,W) prediction.

    Args:
      z: (B, L, D)
      coords_grid: (H, W, 2) or (B, H, W, 2)
    Returns:
      u_hat: (B, 4, H, W) channels-first
    """
    if coords_grid.ndim == 3:
        coords_b = jnp.broadcast_to(coords_grid[None, ...], (z.shape[0],) + coords_grid.shape)
    elif coords_grid.ndim == 4:
        coords_b = coords_grid
    else:
        raise ValueError(f"coords_grid must be (H,W,2) or (B,H,W,2), got shape={coords_grid.shape}")

    b, h, w, _ = coords_b.shape
    coords_flat = coords_b.reshape(b, h * w, 2)  # (B, Nq, 2)
    u_flat = decoder.apply(decoder_params, z, coords_flat)  # (B, Nq, 4)
    u_hw = u_flat.reshape(b, h, w, u_flat.shape[-1])  # (B, H, W, 4)
    u_cf = jnp.transpose(u_hw, (0, 3, 1, 2))  # (B, 4, H, W)
    return u_cf


def loss_fn(
    encoder,
    decoder,
    params: Tuple[Any, Any],
    batch: Dict[str, jnp.ndarray],
    *,
    loss_cfg,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Gyro autoencoder loss.

    Expected batch keys (flexible, but these are the intended ones):
      - x: (B, 4, H, W) or (B, H, W, 4) input
      - y: (B, 4, H, W) or (B, H, W, 4) ground truth
      - coords_grid: (H, W, 2) or (B, H, W, 2) full grid coords for decoding and spectral loss

    Note: you can train with subsets of queries later; for the spectrum/cross losses as written,
    you want full-grid `coords_grid` + full-grid `y`.
    """
    encoder_params, decoder_params = params
    x = batch["x"]
    y = batch["y"]
    coords_grid = batch.get("coords_grid", None)

    z = encoder.apply(encoder_params, x)  # (B, L, D)

    if coords_grid is None:
        raise ValueError("batch must include 'coords_grid' for full-grid decoding/losses.")
    u_hat = decode_full_grid(decoder, decoder_params, z, coords_grid)  # (B,4,H,W)

    total, metrics = compute_gyro_ae_losses(
        u_hat=u_hat,
        u_gt=y,
        coords_grid=coords_grid,
        rec_kind=loss_cfg.rec.kind,
        rec_huber_delta=loss_cfg.rec.huber_delta,
        w_rec=loss_cfg.rec.weight,
        w_spec=loss_cfg.spec.weight if loss_cfg.spec.enabled else 0.0,
        w_cross=loss_cfg.cross.weight if loss_cfg.cross.enabled else 0.0,
        spec_num_bins=loss_cfg.spec.num_bins,
        spec_eps=loss_cfg.spec.eps,
    )

    return total, metrics


def create_train_step(encoder, decoder, mesh, *, loss_cfg):
    """Create a sharded train step (mirrors burgers/diffusion/model_utils.py style)."""

    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P(), P(), P(), P()),
        check_rep=False,
    )
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(
            partial(loss_fn, encoder, decoder, loss_cfg=loss_cfg),
            has_aux=True,
        )
        (loss, metrics), grads = grad_fn(state.params, batch)

        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        loss_rec = lax.pmean(metrics["loss_rec"], "batch")
        loss_spec = lax.pmean(metrics["loss_spec"], "batch")
        loss_cross = lax.pmean(metrics["loss_cross"], "batch")

        state = state.apply_gradients(grads=grads)
        return state, loss, loss_rec, loss_spec, loss_cross

    return train_step


