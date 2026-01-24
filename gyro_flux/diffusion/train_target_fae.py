"""Training script for Target FAE (CGYRO time-series).

Trains the time-series encoder/decoder to reconstruct
CGYRO flux history at arbitrary timesteps.

Handles variable-length sequences (T = ~150 to ~3000 timesteps).

Usage:
    python train_target_fae.py --config=configs/target_fae.py --data_path=/path/to/data
"""

import os
import json
import time

# Disable command buffers to avoid OOM with variable-length sequences
# Each unique sequence length triggers a new JAX compilation, creating many CUDA graphs
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import ml_collections
from absl import app, flags
import wandb

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from flax.training import train_state

import optax
import numpy as np
from sklearn.model_selection import train_test_split

from gyro_flux.models import TimeSeriesEncoder, ContinuousTimeDecoder
from gyro_flux.data_utils import CGYRODataset, create_cgyro_dataloader, get_paired_folders
from gyro_flux.diffusion.model_utils import (
    create_target_train_step,
    create_target_eval_step,
    prepare_target_batch,
    target_loss_fn,
)

from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to config file.")
flags.DEFINE_string("data_path", None, "Path to data directory (overrides config).")
flags.DEFINE_string("workdir", None, "Working directory for checkpoints.")


# =============================================================================
# Utilities
# =============================================================================

def create_optimizer(config):
    """Create learning rate schedule and optimizer."""
    lr = optax.warmup_exponential_decay_schedule(
        init_value=config.lr.init_value,
        peak_value=config.lr.peak_value,
        warmup_steps=config.lr.warmup_steps,
        transition_steps=config.lr.transition_steps,
        decay_rate=config.lr.decay_rate,
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.optim.clip_norm),
        optax.adamw(lr, weight_decay=config.optim.weight_decay),
    )
    return lr, tx


def create_target_fae_state(config, encoder, decoder, tx):
    """Initialize encoder and decoder parameters.
    
    Note: For variable-length input, we use a fixed dummy size for init.
    The model now uses explicit time coordinates instead of positional embeddings.
    """
    # Dummy inputs for initialization
    x = jnp.ones(config.x_dim)  # (B, T_dummy, 2)
    t = jnp.ones(config.t_dim)  # (B, T_dummy, 1) normalized time coordinates
    t_query = jnp.ones(config.t_query_dim)  # (B, N_q, 1) normalized time queries

    encoder_params = encoder.init(random.PRNGKey(config.seed), x, t)
    z = encoder.apply(encoder_params, x, t)  # (B, num_latents, emb_dim)
    decoder_params = decoder.init(random.PRNGKey(config.seed), z, t_query)
    params = (encoder_params, decoder_params)

    state = train_state.TrainState.create(apply_fn=decoder.apply, params=params, tx=tx)
    return state


def compute_total_params(state):
    """Count total parameters in the model."""
    from jax.flatten_util import ravel_pytree
    flatten_params, _ = ravel_pytree(state.params)
    total_params = len(flatten_params)

    if total_params >= 1_000_000_000:
        print(f"Total number of parameters: {total_params / 1_000_000_000:.2f} billion")
    elif total_params >= 1_000_000:
        print(f"Total number of parameters: {total_params / 1_000_000:.2f} million")
    else:
        print(f"Total number of parameters: {total_params / 1_000:.2f} thousand")
    return total_params


# =============================================================================
# Main Training Function
# =============================================================================

def train_and_evaluate(config: ml_collections.ConfigDict):
    """Main training loop.
    
    Note: Uses batch_size=1 for variable-length sequences. Each sample
    triggers a re-compilation due to different sequence lengths, but JAX
    caches compiled functions so this overhead diminishes over training.
    """
    
    # Override data path if provided via flag
    if FLAGS.data_path is not None:
        config.dataset.data_path = FLAGS.data_path
    
    if config.dataset.data_path is None:
        raise ValueError("data_path must be set in config or via --data_path flag")
    
    # Initialize models
    encoder_config = dict(config.model.encoder)
    slice_some_time = config.dataset.slice_some_time
    slice_length = config.dataset.slice_length  # Initial value, may be adjusted by dataset

    # Time-based slicing parameters (new, recommended approach)
    slice_by_time = getattr(config.dataset, 'slice_by_time', False)
    max_tau = getattr(config.dataset, 'max_tau', 0.1)
    
    # Note: encoder will be re-initialized after dataset creation if slice_length was adjusted
    encoder = TimeSeriesEncoder(**encoder_config)
    decoder = ContinuousTimeDecoder(**config.model.decoder)
    
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state (will be recreated if encoder is re-initialized)
    state = create_target_fae_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create dataset split (80/10/10: train/val/test)
    # Use same seed for reproducibility
    all_folders = get_paired_folders(config.dataset.data_path)
    print(f"Total folders found: {len(all_folders)}")
    
    # First split: 80% train, 20% temp (for val + test)
    train_folders, temp_folders = train_test_split(
        all_folders, test_size=0.2, random_state=config.seed
    )
    
    # Second split: 50% of temp = 10% of total for val, 10% for test
    val_folders, test_folders = train_test_split(
        temp_folders, test_size=0.5, random_state=config.seed
    )
    
    print(f"Train: {len(train_folders)} samples ({100*len(train_folders)/len(all_folders):.1f}%)")
    print(f"Val: {len(val_folders)} samples ({100*len(val_folders)/len(all_folders):.1f}%)")
    print(f"Test: {len(test_folders)} samples ({100*len(test_folders)/len(all_folders):.1f}%)")
    
    # Create datasets
    use_padded = config.dataset.use_padded_sequences
    max_seq_length = config.dataset.max_seq_length

    # Normalization strategy
    normalize_per_sample = getattr(config.dataset, 'normalize_per_sample', False)
    normalize_global = getattr(config.dataset, 'normalize_global', False)

    if normalize_per_sample:
        # Per-sample normalization: each sample normalized independently
        # This addresses the 324x magnitude variation across files
        print("Using per-sample normalization (recommended for heterogeneous data)")
        train_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=False,
            normalize_per_sample=True,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=train_folders,
        )

        val_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=False,
            normalize_per_sample=True,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=val_folders,
        )
        train_stats = None  # Not used in per-sample mode

    elif normalize_global:
        # Global normalization: compute stats from training set
        print("Using global normalization")
        train_dataset_for_stats = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=False,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=train_folders,
        )
        # Compute stats manually
        all_data = np.concatenate(train_dataset_for_stats.data, axis=0)
        train_mean = all_data.mean(axis=0, keepdims=True)
        train_std = all_data.std(axis=0, keepdims=True) + 1e-8
        train_stats = (train_mean, train_std)

        # Create train dataset with global normalization
        train_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=True,
            stats=train_stats,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=train_folders,
        )

        # Create val dataset with same global stats
        val_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=True,
            stats=train_stats,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=val_folders,
        )

    else:
        # No normalization
        print("No normalization applied")
        train_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=False,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=train_folders,
        )

        val_dataset = CGYRODataset(
            data_path=config.dataset.data_path,
            normalize=False,
            use_padding=use_padded,
            max_seq_length=max_seq_length,
            slice_some_time=slice_some_time,
            slice_length=slice_length,
            slice_by_time=slice_by_time,
            max_tau=max_tau,
            folder_list=val_folders,
        )
        train_stats = None
    
    # Get actual sequence length from dataset (may have been adjusted)
    if slice_by_time:
        # Time-based slicing: sequences have variable lengths, padded to max
        # The padded_data has shape (N, pad_length, 2)
        actual_pad_length = train_dataset.padded_data.shape[1]
        print(f"Time-based slicing: Sequences padded to {actual_pad_length} (from max_tau={max_tau})")
        print(f"  Length range in dataset: [{train_dataset.lengths.min()}, {train_dataset.lengths.max()}]")

        # Update config for model initialization with padded length
        config.x_dim = [2, actual_pad_length, 2]
        config.t_dim = [2, actual_pad_length, 1]

        # Re-initialize encoder with correct dimensions
        encoder = TimeSeriesEncoder(**encoder_config)
        state = create_target_fae_state(config, encoder, decoder, tx)
        print(f"Time-based slicing: Re-initialized model with padded length={actual_pad_length}")

        # For batch size logic later
        slice_length = actual_pad_length

    elif slice_some_time:
        actual_slice_length = int(train_dataset.slice_length)  # Convert numpy int64 to Python int
        if actual_slice_length != slice_length:
            print(f"Slice length adjusted: {slice_length} -> {actual_slice_length} (based on minimum sequence length)")
            slice_length = actual_slice_length
            # Update config for model initialization
            config.x_dim = [2, slice_length, 2]
            config.t_dim = [2, slice_length, 1]
            # Re-initialize encoder (no changes needed to encoder_config, just re-init with same config)
            encoder = TimeSeriesEncoder(**encoder_config)
            # Re-create train state with updated encoder
            state = create_target_fae_state(config, encoder, decoder, tx)
            print(f"Slice mode: Re-initialized model with adjusted slice_length={slice_length}")
        else:
            print(f"Slice mode: Using fixed sequence length {slice_length}")
    
    # Now finalize state sharding and train step (after potential encoder re-initialization)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")
    
    # Create sharding for data parallelism
    # Note: For batch_size=1, we replicate across devices rather than shard
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create train and eval step functions
    use_time_weighting = config.training.use_time_weighting
    growth_phase_weight = config.training.growth_phase_weight
    use_relative_l2 = getattr(config.training, 'use_relative_l2', False)
    rel_l2_eps = getattr(config.training, 'rel_l2_eps', 1.0)

    train_step = create_target_train_step(
        encoder, decoder, mesh,
        use_time_weighting=use_time_weighting,
        growth_phase_weight=growth_phase_weight,
        use_relative_l2=use_relative_l2,
        rel_l2_eps=rel_l2_eps,
    )
    
    eval_step = create_target_eval_step(encoder, decoder, mesh)
    
    # batch_size logic:
    # - slice_by_time: variable lengths padded to max, can use larger batch size
    # - slice_some_time: fixed length, can use larger batch size
    # - use_padded: fixed length, can use larger batch size
    # - otherwise: variable length, must use batch_size=1
    if slice_by_time:
        batch_size = config.dataset.train_batch_size
        print(f"Time-based slicing: Using batch_size={batch_size} (padded sequences)")
    elif slice_some_time:
        batch_size = config.dataset.train_batch_size  # Use config batch size (can be > 1)
        print(f"Slice mode: Using batch_size={batch_size} (fixed-length sequences)")
    elif use_padded:
        batch_size = config.dataset.train_batch_size
    else:
        batch_size = 1  # Variable length requires batch_size=1
    
    train_loader = create_cgyro_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
    )
    
    val_loader = create_cgyro_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,  # No shuffling for validation
    )

    # Working directory for checkpoints
    # Structure: /home/guzmans/fundiff/{model_name}/{run_name}/
    if FLAGS.workdir is not None:
        workdir = FLAGS.workdir
    else:
        model_name = config.model.model_name
        run_name = config.wandb.run_name or "default"
        workdir = os.path.join(os.getcwd(), model_name, run_name)
    
    ckpt_path = os.path.join(workdir, "ckpt")
    
    # Check if W&B is enabled
    use_wandb = config.wandb.use_wandb
    
    if jax.process_index() == 0:
        os.makedirs(ckpt_path, exist_ok=True)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(workdir, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B (if enabled)
        if use_wandb:
            wandb_config = config.wandb
            run_name = getattr(wandb_config, "run_name", None) or config.model.model_name
            wandb.init(
                project=wandb_config.project,
                entity=getattr(wandb_config, "entity", None),
                group=getattr(wandb_config, "group", None),
                name=run_name,
                notes=getattr(wandb_config, "notes", None),  # Description shown in W&B UI
                tags=getattr(wandb_config, "tag", None).split(',') if getattr(wandb_config, "tag", None) else None,
                config=config.to_dict(),
            )
            wandb.log({"num_params": num_params}, step=0)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Training loop
    rng_key = jax.random.PRNGKey(config.seed)
    step = 0
    num_queries = config.training.num_queries
    
    print(f"Starting training for {config.training.max_steps} steps...")
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Val dataset size: {len(val_dataset)} samples")
    print(f"Timestep range: [{train_dataset.lengths.min()}, {train_dataset.lengths.max()}]")
    print(f"Using time weighting: {use_time_weighting}")
    print(f"Using padded sequences: {use_padded}")
    print(f"Using per-sample normalization: {normalize_per_sample}")
    print(f"Using relative L2 loss: {use_relative_l2}")
    if use_relative_l2:
        print(f"  rel_l2_eps: {rel_l2_eps}")
    if slice_by_time:
        print(f"Time-based slicing: max_tau={max_tau} (t ∈ [3.0, {3.0 + max_tau * 1000.0}])")
        print(f"  Padded length: {train_dataset.padded_data.shape[1]}")
        print(f"  Length range: [{train_dataset.lengths.min()}, {train_dataset.lengths.max()}]")
    elif slice_some_time:
        print(f"Slice mode: Fixed length {slice_length} timesteps")
    if use_time_weighting:
        print(f"Growth phase weight: {growth_phase_weight}")
    if use_padded or slice_some_time or slice_by_time:
        print(f"Batch size: {batch_size}")
    
    epoch = 0
    train_loss_val = 0.0  # Track last training loss for epoch summary
    while step < config.training.max_steps:
        epoch += 1
        start_time = time.time()
        
        for batch_data in train_loader:
            if step >= config.training.max_steps:
                break
            
            rng_key, subkey = jax.random.split(rng_key)
            
            # Extract CGYRO data and time from collate output
            cgyro = jnp.array(batch_data['cgyro'])  # (B, T, 2)
            time_coords = jnp.array(batch_data['time'])     # (B, T, 1) normalized time coordinates
            lengths = batch_data['lengths']  # (B,) original lengths or slice_length if sliced
            
            if slice_by_time:
                # Time-based slicing: variable lengths padded to max
                # Use min of actual lengths in batch to ensure queries are in valid range
                min_len = int(lengths.min())
                actual_num_queries = min(num_queries, min_len)
                seq_len = cgyro.shape[1]  # Padded length for logging
                max_query_idx = min_len
            elif slice_some_time:
                # Slice mode: all sequences have fixed length (slice_length)
                seq_len = slice_length
                actual_num_queries = min(num_queries, slice_length)
                max_query_idx = slice_length  # All sequences are same length
            elif use_padded:
                # Padded mode: all sequences same length, use min of original lengths for queries
                # This ensures we only query valid (non-padded) timesteps
                min_len = int(lengths.min())
                actual_num_queries = min(num_queries, min_len)
                seq_len = max_seq_length  # For logging
                max_query_idx = min_len
            else:
                # Variable length mode: batch_size=1
                seq_len = cgyro.shape[1]
                actual_num_queries = min(num_queries, seq_len)
                max_query_idx = None
            
            # Prepare batch with random query sampling
            # Note: queries sampled within valid range of shortest sequence in batch
            # prepare_target_batch now extracts time values at query indices
            batch = prepare_target_batch(cgyro, time_coords, actual_num_queries, subkey, 
                                         max_query_idx=max_query_idx)
            
            # Shard batch across devices
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )
            
            # Train step
            state, loss, loss_recon = train_step(state, batch)
            step = int(state.step)
            train_loss_val = loss.item()  # Track for epoch summary
            
            # Logging
            if step % config.logging.log_interval == 0:
                loss_recon_val = loss_recon.item()
                current_lr = lr(step)

                if jax.process_index() == 0:
                    print(f"step: {step}, loss: {train_loss_val:.3e}, seq_len: {seq_len}, lr: {current_lr:.3e}")

                    # === DIAGNOSTICS: Log padding and query distribution ===
                    batch_lengths = np.array(lengths)
                    min_len = int(batch_lengths.min())
                    max_len = int(batch_lengths.max())
                    mean_len = float(batch_lengths.mean())
                    pad_length = cgyro.shape[1]
                    padding_ratio = 1.0 - (mean_len / pad_length)

                    # Query time distribution
                    t_query_np = np.array(batch['t_query'])  # (B, N_q, 1)
                    mean_query_tau = float(t_query_np.mean())
                    early_phase_ratio = float((t_query_np < 0.03).mean())
                    late_phase_ratio = float((t_query_np > 0.07).mean())

                    if use_wandb:
                        log_dict = {
                            "loss/train": train_loss_val,
                            "loss_recon/train": loss_recon_val,
                            "lr": current_lr,
                            "epoch": epoch,
                            "seq_len": seq_len,
                            # Diagnostics
                            "diag/min_seq_len": min_len,
                            "diag/max_seq_len": max_len,
                            "diag/mean_seq_len": mean_len,
                            "diag/padding_ratio": padding_ratio,
                            "diag/mean_query_tau": mean_query_tau,
                            "diag/early_phase_ratio": early_phase_ratio,
                            "diag/late_phase_ratio": late_phase_ratio,
                        }
                        wandb.log(log_dict, step=step)
                    else:
                        print(f"  DIAG: len=[{min_len},{max_len}], pad={padding_ratio:.2f}, "
                              f"early={early_phase_ratio:.2f}, late={late_phase_ratio:.2f}")
            
            # Save checkpoint
            if step % config.saving.save_interval == 0:
                save_checkpoint(ckpt_mngr, state)
        
        # Validation at end of each epoch
        if jax.process_index() == 0:
            print(f"Epoch {epoch} completed, running validation...")
        
        val_loss_sum = 0.0
        val_loss_recon_sum = 0.0
        val_batches = 0
        val_rng_key = jax.random.PRNGKey(config.seed + epoch)  # Deterministic but different per epoch
        
        for val_batch_data in val_loader:
            val_rng_key, val_subkey = jax.random.split(val_rng_key)
            
            cgyro_val = jnp.array(val_batch_data['cgyro'])
            time_coords_val = jnp.array(val_batch_data['time'])
            lengths_val = val_batch_data['lengths']
            
            if slice_by_time:
                # Time-based slicing: use min of actual lengths in batch
                min_len = int(lengths_val.min())
                actual_num_queries = min(num_queries, min_len)
                max_query_idx = min_len
            elif slice_some_time:
                actual_num_queries = min(num_queries, slice_length)
                max_query_idx = slice_length
            elif use_padded:
                min_len = int(lengths_val.min())
                actual_num_queries = min(num_queries, min_len)
                max_query_idx = min_len
            else:
                seq_len_val = cgyro_val.shape[1]
                actual_num_queries = min(num_queries, seq_len_val)
                max_query_idx = None
            
            val_batch = prepare_target_batch(
                cgyro_val, time_coords_val, actual_num_queries, val_subkey,
                max_query_idx=max_query_idx
            )
            
            # Shard batch across devices
            val_batch = multihost_utils.host_local_array_to_global_array(
                val_batch, mesh, P("batch")
            )
            
            # Compute validation loss (no gradients)
            val_loss, val_metrics = target_loss_fn(
                encoder, decoder, state.params, val_batch,
                use_time_weighting=use_time_weighting,
                growth_phase_weight=growth_phase_weight,
                use_relative_l2=use_relative_l2,
                rel_l2_eps=rel_l2_eps,
            )
            
            val_loss_sum += val_loss.item()
            val_loss_recon_sum += val_metrics['loss_recon'].item()
            val_batches += 1
        
        # Average validation metrics
        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
        avg_val_loss_recon = val_loss_recon_sum / val_batches if val_batches > 0 else 0.0
        
        end_time = time.time()
        if jax.process_index() == 0:
            print(f"Epoch {epoch} completed in {end_time - start_time:.2f}s")
            print(f"  Train loss: {train_loss_val:.3e}, Val loss: {avg_val_loss:.3e}")

        # Log val metrics at same interval as train metrics
        if step % config.logging.log_interval == 0 and jax.process_index() == 0:
            if use_wandb:
                wandb.log({
                    "loss/val": avg_val_loss,
                    "loss_recon/val": avg_val_loss_recon,
                    "epoch": epoch,
                }, step=step)

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()
    
    if jax.process_index() == 0 and use_wandb:
        wandb.finish()


def main(argv):
    del argv  # Unused
    
    # Import config
    if FLAGS.config is None:
        # Use default config
        from gyro_flux.diffusion.configs.target_fae import get_config
        config = get_config()
    else:
        # Load config from file
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", FLAGS.config)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.get_config()
    
    train_and_evaluate(config)


if __name__ == "__main__":
    # data_path can be set in config or via --data_path flag
    app.run(main)

