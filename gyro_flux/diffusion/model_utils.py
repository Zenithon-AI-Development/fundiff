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
    use_relative_l2: bool = False,
    rel_l2_eps: float = 1.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Reconstruction loss for Target (CGYRO) FAE.

    Args:
        encoder: TimeSeriesEncoder module
        decoder: ContinuousTimeDecoder module
        params: (encoder_params, decoder_params)
        batch: Dictionary with:
            - 'cgyro': (B, T, 2) input flux history (T may vary, but batched same T)
            - 'time': (B, T, 1) normalized time coordinates for input sequence
            - 't_query': (B, N_q, 1) normalized time coordinates to reconstruct
            - 'targets': (B, N_q, 2) ground truth flux at query times
        use_time_weighting: If True, weight early timesteps more heavily
        growth_phase_weight: Weight multiplier for growth phase (τ < 0.03)
        use_relative_l2: If True, use relative L2 loss (scale-invariant)
        rel_l2_eps: Epsilon for relative L2 denominator stability

    Returns:
        loss: scalar loss value
        metrics: dict with individual loss components
    """
    encoder_params, decoder_params = params

    cgyro = batch['cgyro']      # (B, T, 2)
    time = batch['time']        # (B, T, 1)
    t_query = batch['t_query']  # (B, N_q, 1)
    targets = batch['targets']  # (B, N_q, 2)

    # Encode (with time coordinates)
    z = encoder.apply(encoder_params, cgyro, time)  # (B, num_latents, emb_dim)

    # Decode at query times
    recon = decoder.apply(decoder_params, z, t_query)  # (B, N_q, 2)

    # Per-point squared error
    sq_error = (recon - targets) ** 2  # (B, N_q, 2)

    if use_relative_l2:
        # Relative L2: normalize by target magnitude (per-sample)
        # Compute per-sample normalization factor: mean of target² across time and channels
        target_sq_norm = jnp.mean(targets ** 2, axis=(1, 2), keepdims=True)  # (B, 1, 1)
        # Add epsilon for numerical stability (eps=1.0 acts as soft floor)
        rel_weights = 1.0 / (target_sq_norm + rel_l2_eps)  # (B, 1, 1)
        sq_error = sq_error * rel_weights  # Scale-invariant error

    if use_time_weighting:
        # Weight early timesteps (growth phase) more heavily
        # t_query is already normalized τ, shape (B, N_q, 1)
        tau = t_query.squeeze(-1)  # (B, N_q)
        weights = jnp.where(tau < 0.03, growth_phase_weight, 1.0)  # Growth phase at τ < 0.03
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
    use_relative_l2: bool = False,
    rel_l2_eps: float = 1.0,
):
    """Create sharded train step for Target FAE.

    Args:
        encoder: TimeSeriesEncoder module
        decoder: ContinuousTimeDecoder module
        mesh: JAX device mesh for sharding
        use_time_weighting: If True, weight early timesteps more heavily
        growth_phase_weight: Weight multiplier for growth phase
        use_relative_l2: If True, use relative L2 loss (scale-invariant)
        rel_l2_eps: Epsilon for relative L2 denominator stability

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
                use_relative_l2=use_relative_l2,
                rel_l2_eps=rel_l2_eps,
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
    def encoder_step(encoder_params, cgyro, time):
        z = encoder.apply(encoder_params, cgyro, time)
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
    """Create sharded eval step for Target FAE.
    
    Supports both in-distribution evaluation (with targets) and OOD testing (without targets).
    """
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
        time = batch['time']
        t_query = batch['t_query']
        
        z = encoder.apply(encoder_params, cgyro, time)
        recon = decoder.apply(decoder_params, z, t_query)
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
    time: jnp.ndarray,
    num_queries: int,
    rng_key: jnp.ndarray,
    max_query_idx: int = None,
) -> Dict[str, jnp.ndarray]:
    """Prepare a training batch for Target FAE.
    
    Samples random timestep indices, then extracts their corresponding time values.
    Returns continuous time queries instead of step indices.
    
    Args:
        cgyro: (B, T, 2) flux history
        time: (B, T, 1) normalized time coordinates
        num_queries: Number of timesteps to query
        rng_key: PRNG key for random sampling
        max_query_idx: Optional max index for queries (for padded mode, use shortest seq in batch)
    
    Returns:
        batch dict with 'cgyro', 'time', 't_query', 'targets'
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
    
    # Extract time values at query positions
    # time is (B, T, 1), step_indices is (N_q,)
    # Use advanced indexing to get (B, N_q, 1)
    t_query = time[:, step_indices, :]  # (B, N_q, 1)
    
    return {
        'cgyro': cgyro,
        'time': time,
        't_query': t_query,
        'targets': targets,
    }


def prepare_target_batch_arbitrary_times(
    cgyro: jnp.ndarray,
    time: jnp.ndarray,
    t_query: jnp.ndarray,
    interpolate_targets: bool = True,
) -> Dict[str, jnp.ndarray]:
    """Prepare a validation batch for Target FAE with arbitrary time queries.
    
    Allows querying at arbitrary time values (e.g., higher resolution, OOD testing).
    Optionally interpolates ground truth targets at query times.
    
    Args:
        cgyro: (B, T, 2) flux history
        time: (B, T, 1) normalized time coordinates from dataset
        t_query: (B, N_q, 1) or (N_q, 1) arbitrary normalized time queries
        interpolate_targets: If True, linearly interpolate targets at query times.
                           If False, targets will be None (for OOD testing).
    
    Returns:
        batch dict with 'cgyro', 'time', 't_query', 'targets' (or None if not interpolating)
    
    Example:
        # Higher resolution evaluation: query at 1000 evenly spaced times
        t_min, t_max = time[0, 0, 0], time[0, -1, 0]
        t_query = jnp.linspace(t_min, t_max, 1000)[:, None]  # (1000, 1)
        batch = prepare_target_batch_arbitrary_times(cgyro, time, t_query, interpolate_targets=True)
        
        # OOD testing: query beyond training time range
        t_query_ood = jnp.linspace(t_max, t_max + 0.1, 100)[:, None]  # (100, 1)
        batch_ood = prepare_target_batch_arbitrary_times(cgyro, time, t_query_ood, interpolate_targets=False)
    """
    b, num_steps, num_channels = cgyro.shape
    
    # Ensure t_query has correct shape: (B, N_q, 1)
    if t_query.ndim == 2:
        # (N_q, 1) -> (B, N_q, 1)
        t_query = jnp.broadcast_to(t_query[None, :, :], (b, t_query.shape[0], 1))
    elif t_query.ndim == 1:
        # (N_q,) -> (B, N_q, 1)
        t_query = jnp.broadcast_to(t_query[None, :, None], (b, t_query.shape[0], 1))
    
    n_q = t_query.shape[1]
    
    if interpolate_targets:
        # Linearly interpolate targets at query times
        # time is (B, T, 1), cgyro is (B, T, 2), t_query is (B, N_q, 1)
        time_flat = time[:, :, 0]  # (B, T)
        t_query_flat = t_query[:, :, 0]  # (B, N_q)
        
        # Vectorized interpolation using vmap
        def interpolate_single(time_seq, flux_seq, t_q):
            """Interpolate flux at time t_q given time_seq and flux_seq."""
            # Find insertion index
            idx = jnp.searchsorted(time_seq, t_q)
            
            # Handle boundary cases
            idx = jnp.clip(idx, 1, num_steps - 1)  # Ensure idx-1 and idx are valid
            
            # Get neighboring times and fluxes
            t_low = time_seq[idx - 1]
            t_high = time_seq[idx]
            flux_low = flux_seq[idx - 1, :]  # (2,)
            flux_high = flux_seq[idx, :]  # (2,)
            
            # Linear interpolation
            alpha = (t_q - t_low) / (t_high - t_low + 1e-8)
            alpha = jnp.clip(alpha, 0.0, 1.0)  # Clamp to [0, 1]
            
            return (1 - alpha) * flux_low + alpha * flux_high
        
        # Vectorize over batch and queries
        interpolate_batch = jax.vmap(
            jax.vmap(interpolate_single, in_axes=(None, None, 0)),  # Over queries
            in_axes=(0, 0, 0)  # Over batch
        )
        
        targets = interpolate_batch(time_flat, cgyro, t_query_flat)  # (B, N_q, 2)
    else:
        # No targets (for OOD testing)
        targets = None
    
    return {
        'cgyro': cgyro,
        'time': time,
        't_query': t_query,
        'targets': targets,
    }


# =============================================================================
# DiT Loss and Train Step (Rectified Flow)
# =============================================================================

def dit_loss_fn(
    model,
    params: Any,
    batch: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Rectified Flow loss for FluxDiT.
    
    The model predicts velocity v_θ(z_t, t, z_c) where:
    - z_t = (1 - t) * z_0 + t * z_1  (interpolated between noise and data)
    - Target velocity = z_1 - z_0
    
    Loss: ||v_θ(z_t, t, z_c) - (z_1 - z_0)||²
    
    Args:
        model: FluxDiT module
        params: Model parameters
        batch: Dictionary with:
            - 'z_t': (B, num_latents, emb_dim) interpolated latent
            - 't': (B,) timestep in [0, 1]
            - 'z_c': (B, z_c_dim) TGLF condition
            - 'z_0': (B, num_latents, emb_dim) noise (for target computation)
            - 'z_1': (B, num_latents, emb_dim) data latent (for target computation)
    
    Returns:
        loss: scalar loss value
        metrics: dict with loss components
    """
    z_t = batch['z_t']  # (B, num_latents, emb_dim)
    t = batch['t']      # (B,)
    z_c = batch['z_c']  # (B, z_c_dim)
    z_0 = batch['z_0']  # (B, num_latents, emb_dim)
    z_1 = batch['z_1']  # (B, num_latents, emb_dim)
    
    # Predict velocity
    v_pred = model.apply(params, z_t, t, z_c)  # (B, num_latents, emb_dim)
    
    # Target velocity: z_1 - z_0
    v_target = z_1 - z_0  # (B, num_latents, emb_dim)
    
    # MSE loss
    loss = jnp.mean((v_pred - v_target) ** 2)
    
    metrics = {
        'loss': loss,
    }
    
    return loss, metrics


def create_dit_train_step(model, mesh):
    """Create sharded train step for FluxDiT.
    
    Args:
        model: FluxDiT module
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
            partial(dit_loss_fn, model),
            has_aux=True,
        )
        (loss, metrics), grads = grad_fn(state.params, batch)
        
        # Average gradients across devices
        grads = lax.pmean(grads, "batch")
        loss = lax.pmean(loss, "batch")
        
        state = state.apply_gradients(grads=grads)
        return state, loss, loss
    
    return train_step


def prepare_dit_batch(
    z_0: jnp.ndarray,
    z_1: jnp.ndarray,
    z_c: jnp.ndarray,
    rng_key: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Prepare a training batch for FluxDiT (Rectified Flow).
    
    Samples random timesteps t ~ U(0, 1) and creates interpolated latents z_t.
    
    Args:
        z_0: (B, num_latents, emb_dim) noise latents
        z_1: (B, num_latents, emb_dim) data latents
        z_c: (B, z_c_dim) TGLF conditioning vectors
        rng_key: PRNG key for sampling timesteps
    
    Returns:
        batch dict with 'z_t', 't', 'z_c', 'z_0', 'z_1'
    """
    b = z_0.shape[0]
    
    # Sample random timesteps t ~ U(0, 1)
    t = random.uniform(rng_key, shape=(b,), minval=0.0, maxval=1.0)  # (B,)
    
    # Interpolate: z_t = (1 - t) * z_0 + t * z_1
    t_expanded = t[:, None, None]  # (B, 1, 1) for broadcasting
    z_t = (1.0 - t_expanded) * z_0 + t_expanded * z_1  # (B, num_latents, emb_dim)
    
    return {
        'z_t': z_t,
        't': t,
        'z_c': z_c,
        'z_0': z_0,
        'z_1': z_1,
    }

