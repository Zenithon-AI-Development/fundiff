"""Training utilities for Gyro-Flux FAEs.

Contains loss functions and train step creators for:
- TGLF FAE (fixed input shape)
- Target FAE (variable-length input)
"""

from functools import partial
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


# =============================================================================
# TGLF FAE Loss and Train Step
# =============================================================================

def tglf_loss_fn(
    encoder,
    decoder,
    params: Tuple[Any, Any],
    batch: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Reconstruction loss for TGLF FAE.
    
    The encoder outputs z_c directly (conditioning vector), and the decoder
    reconstructs from z_c. This ensures z_c is directly optimized.
    
    Args:
        encoder: TGLFEncoder module (outputs z_c directly)
        decoder: TGLFDecoder module (takes z_c, not z)
        params: (encoder_params, decoder_params)
        batch: Dictionary with:
            - 'tglf': (B, 21, 108) input TGLF spectra
            - 'ky_query': (B, N_q, 1) ky indices to reconstruct
            - 'targets': (B, N_q, 108) ground truth at query positions
    
    Returns:
        loss: scalar loss value
        metrics: dict with individual loss components
    """
    encoder_params, decoder_params = params
    
    tglf = batch['tglf']           # (B, 21, 108)
    ky_query = batch['ky_query']   # (B, N_q, 1)
    targets = batch['targets']     # (B, N_q, 108)
    
    # Encode to z_c directly
    z_c = encoder.apply(encoder_params, tglf)  # (B, z_c_dim)
    
    # Decode at query positions from z_c
    recon = decoder.apply(decoder_params, z_c, ky_query)  # (B, N_q, 108)
    
    # MSE loss
    loss_recon = jnp.mean((recon - targets) ** 2)
    
    metrics = {
        'loss_recon': loss_recon,
    }
    
    return loss_recon, metrics


def create_tglf_train_step(encoder, decoder, mesh):
    """Create sharded train step for TGLF FAE.
    
    Args:
        encoder: TGLFEncoder module
        decoder: TGLFDecoder module
        mesh: JAX device mesh for sharding
    
    Returns:
        train_step function
    """
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P(), P()),
        check_rep=False,
    )
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(
            partial(tglf_loss_fn, encoder, decoder),
            has_aux=True,
        )
        (loss, metrics), grads = grad_fn(state.params, batch)
        
        # Average gradients across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        loss_recon = lax.pmean(metrics['loss_recon'], "batch")
        
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_recon
    
    return train_step


# =============================================================================
# Target FAE Loss and Train Step
# =============================================================================

def target_loss_fn(
    encoder,
    decoder,
    params: Tuple[Any, Any],
    batch: Dict[str, jnp.ndarray],
    use_time_weighting: bool = False,
    growth_phase_weight: float = 5.0,
    time_scale: float = 1000.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Reconstruction loss for Target (CGYRO) FAE.
    
    Args:
        encoder: TimeSeriesEncoder module
        decoder: ContinuousTimeDecoder module
        params: (encoder_params, decoder_params)
        batch: Dictionary with:
            - 'cgyro': (B, T, 2) input flux history (T may vary, but batched same T)
            - 'step_query': (B, N_q) step indices to reconstruct
            - 'targets': (B, N_q, 2) ground truth flux at query steps
        use_time_weighting: If True, weight early timesteps more heavily
        growth_phase_weight: Weight multiplier for growth phase (τ < 0.03)
        time_scale: Normalization for τ = step / time_scale
    
    Returns:
        loss: scalar loss value
        metrics: dict with individual loss components
    """
    encoder_params, decoder_params = params
    
    cgyro = batch['cgyro']           # (B, T, 2)
    step_query = batch['step_query'] # (B, N_q)
    targets = batch['targets']       # (B, N_q, 2)
    
    # Encode
    z = encoder.apply(encoder_params, cgyro)  # (B, num_latents, emb_dim)
    
    # Decode at query steps
    recon = decoder.apply(decoder_params, z, step_query)  # (B, N_q, 2)
    
    # Per-point squared error
    sq_error = (recon - targets) ** 2  # (B, N_q, 2)
    
    if use_time_weighting:
        # Weight early timesteps (growth phase) more heavily
        tau = step_query.astype(jnp.float32) / time_scale  # (B, N_q)
        weights = jnp.where(tau < 0.03, growth_phase_weight, 1.0)  # Growth phase ~30 steps
        weights = weights[..., None]  # (B, N_q, 1)
        loss_recon = jnp.mean(sq_error * weights)
    else:
        loss_recon = jnp.mean(sq_error)
    
    metrics = {
        'loss_recon': loss_recon,
    }
    
    return loss_recon, metrics


def create_target_train_step(
    encoder,
    decoder,
    mesh,
    use_time_weighting: bool = False,
    growth_phase_weight: float = 5.0,
    time_scale: float = 1000.0,
):
    """Create sharded train step for Target FAE.
    
    Args:
        encoder: TimeSeriesEncoder module
        decoder: ContinuousTimeDecoder module
        mesh: JAX device mesh for sharding
        use_time_weighting: If True, weight early timesteps more heavily
        growth_phase_weight: Weight multiplier for growth phase
        time_scale: Normalization for τ = step / time_scale
    
    Returns:
        train_step function
    """
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=(P(), P(), P()),
        check_rep=False,
    )
    def train_step(state, batch):
        grad_fn = jax.value_and_grad(
            partial(
                target_loss_fn,
                encoder,
                decoder,
                use_time_weighting=use_time_weighting,
                growth_phase_weight=growth_phase_weight,
                time_scale=time_scale,
            ),
            has_aux=True,
        )
        (loss, metrics), grads = grad_fn(state.params, batch)
        
        # Average gradients across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        loss_recon = lax.pmean(metrics['loss_recon'], "batch")
        
        state = state.apply_gradients(grads=grads)
        return state, loss, loss_recon
    
    return train_step


# =============================================================================
# Encoder-only steps (for diffusion training)
# =============================================================================

def create_tglf_encoder_step(encoder, mesh):
    """Create sharded encoder step for TGLF (used in diffusion training).
    
    Returns z_c (B, z_c_dim) directly for DiT conditioning.
    """
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        check_rep=False,
    )
    def encoder_step(encoder_params, tglf):
        z_c = encoder.apply(encoder_params, tglf)  # (B, z_c_dim)
        return z_c
    
    return encoder_step


def create_target_encoder_step(encoder, mesh):
    """Create sharded encoder step for Target (used in diffusion training)."""
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        check_rep=False,
    )
    def encoder_step(encoder_params, cgyro):
        z = encoder.apply(encoder_params, cgyro)
        return z
    
    return encoder_step


# =============================================================================
# Evaluation steps
# =============================================================================

def create_tglf_eval_step(encoder, decoder, mesh):
    """Create sharded eval step for TGLF FAE."""
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        check_rep=False,
    )
    def eval_step(params, batch):
        encoder_params, decoder_params = params
        tglf = batch['tglf']
        ky_query = batch['ky_query']
        
        z_c = encoder.apply(encoder_params, tglf)  # (B, z_c_dim)
        recon = decoder.apply(decoder_params, z_c, ky_query)
        return recon
    
    return eval_step


def create_target_eval_step(encoder, decoder, mesh):
    """Create sharded eval step for Target FAE."""
    @jax.jit
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P("batch")),
        out_specs=P("batch"),
        check_rep=False,
    )
    def eval_step(params, batch):
        encoder_params, decoder_params = params
        cgyro = batch['cgyro']
        step_query = batch['step_query']
        
        z = encoder.apply(encoder_params, cgyro)
        recon = decoder.apply(decoder_params, z, step_query)
        return recon
    
    return eval_step


# =============================================================================
# Batch Preparation Utilities
# =============================================================================

def prepare_tglf_batch(
    tglf: jnp.ndarray,
    num_queries: int,
    rng_key: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Prepare a training batch for TGLF FAE.
    
    Samples random ky indices as query points.
    
    Args:
        tglf: (B, 21, 108) TGLF spectra
        num_queries: Number of ky positions to query (capped at num_ky)
        rng_key: PRNG key for random sampling
    
    Returns:
        batch dict with 'tglf', 'ky_query', 'targets'
    """
    b, num_ky, num_features = tglf.shape
    
    # Cap num_queries to available ky modes
    actual_queries = min(num_queries, num_ky)
    
    # Sample random ky indices (without replacement)
    ky_indices = random.choice(
        rng_key,
        num_ky,
        shape=(actual_queries,),
        replace=False,
    )  # (N_q,)
    
    # Get targets at query positions
    targets = tglf[:, ky_indices, :]  # (B, N_q, 108)
    
    # Convert indices to query format
    ky_query = ky_indices[None, :, None]  # (1, N_q, 1)
    ky_query = jnp.broadcast_to(ky_query, (b, actual_queries, 1))  # (B, N_q, 1)
    ky_query = ky_query.astype(jnp.float32)
    
    return {
        'tglf': tglf,
        'ky_query': ky_query,
        'targets': targets,
    }


def prepare_target_batch(
    cgyro: jnp.ndarray,
    num_queries: int,
    rng_key: jnp.ndarray,
    max_query_idx: int = None,
) -> Dict[str, jnp.ndarray]:
    """Prepare a training batch for Target FAE.
    
    Samples random timestep indices as query points.
    
    Args:
        cgyro: (B, T, 2) flux history
        num_queries: Number of timesteps to query
        rng_key: PRNG key for random sampling
        max_query_idx: Optional max index for queries (for padded mode, use shortest seq in batch)
    
    Returns:
        batch dict with 'cgyro', 'step_query', 'targets'
    """
    b, num_steps, num_channels = cgyro.shape
    
    # For padded sequences, only sample from valid (non-padded) range
    query_range = max_query_idx if max_query_idx is not None else num_steps
    
    # Sample random step indices within valid range
    step_indices = random.choice(
        rng_key,
        query_range,
        shape=(num_queries,),
        replace=False,
    )  # (N_q,)
    
    # Get targets at query positions
    targets = cgyro[:, step_indices, :]  # (B, N_q, 2)
    
    # Broadcast step indices to batch
    step_query = jnp.broadcast_to(step_indices[None, :], (b, num_queries))  # (B, N_q)
    
    return {
        'cgyro': cgyro,
        'step_query': step_query,
        'targets': targets,
    }

