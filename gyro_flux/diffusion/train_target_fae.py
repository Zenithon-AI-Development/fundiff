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

from gyro_flux.models import TimeSeriesEncoder, ContinuousTimeDecoder
from gyro_flux.data_utils import CGYRODataset, create_cgyro_dataloader
from gyro_flux.diffusion.model_utils import (
    create_target_train_step,
    prepare_target_batch,
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
    The model handles variable lengths via interpolated positional embeddings.
    """
    # Dummy inputs for initialization
    x = jnp.ones(config.x_dim)  # (B, T_dummy, 2)
    step_query = jnp.ones(config.step_query_dim, dtype=jnp.int32)  # (B, N_q)

    encoder_params = encoder.init(random.PRNGKey(config.seed), x)
    z = encoder.apply(encoder_params, x)  # (B, num_latents, emb_dim)
    decoder_params = decoder.init(random.PRNGKey(config.seed), z, step_query)
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
    # Override use_fixed_pe if using padded sequences (for efficient JAX compilation)
    encoder_config = dict(config.model.encoder)
    if config.dataset.use_padded_sequences:
        encoder_config['use_fixed_pe'] = True
    
    encoder = TimeSeriesEncoder(**encoder_config)
    decoder = ContinuousTimeDecoder(**config.model.decoder)
    
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_target_fae_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    # Note: For batch_size=1, we replicate across devices rather than shard
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create train step function
    use_time_weighting = config.training.use_time_weighting
    growth_phase_weight = config.training.growth_phase_weight
    time_scale = config.model.decoder.time_scale
    
    train_step = create_target_train_step(
        encoder, decoder, mesh,
        use_time_weighting=use_time_weighting,
        growth_phase_weight=growth_phase_weight,
        time_scale=time_scale,
    )

    # Create dataset and dataloader
    use_padded = config.dataset.use_padded_sequences
    max_seq_length = config.dataset.max_seq_length
    
    train_dataset = CGYRODataset(
        data_path=config.dataset.data_path,
        normalize=True,
        use_padding=use_padded,
        max_seq_length=max_seq_length,
    )
    
    # batch_size=1 for variable length, can be larger for padded
    batch_size = config.dataset.train_batch_size if use_padded else 1
    train_loader = create_cgyro_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
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
    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Timestep range: [{train_dataset.lengths.min()}, {train_dataset.lengths.max()}]")
    print(f"Using time weighting: {use_time_weighting}")
    print(f"Using padded sequences: {use_padded}")
    if use_time_weighting:
        print(f"Growth phase weight: {growth_phase_weight}")
    if use_padded:
        print(f"Batch size: {batch_size}, Max seq length: {max_seq_length}")
    
    epoch = 0
    while step < config.training.max_steps:
        epoch += 1
        start_time = time.time()
        
        for batch_data in train_loader:
            if step >= config.training.max_steps:
                break
            
            rng_key, subkey = jax.random.split(rng_key)
            
            # Extract CGYRO data from collate output
            cgyro = jnp.array(batch_data['cgyro'])  # (B, T, 2)
            lengths = batch_data['lengths']  # (B,) original lengths
            
            if use_padded:
                # Padded mode: all sequences same length, use min of original lengths for queries
                # This ensures we only query valid (non-padded) timesteps
                min_len = int(lengths.min())
                actual_num_queries = min(num_queries, min_len)
                seq_len = max_seq_length  # For logging
            else:
                # Variable length mode: batch_size=1
                seq_len = cgyro.shape[1]
                actual_num_queries = min(num_queries, seq_len)
            
            # Prepare batch with random query sampling
            # Note: queries sampled within valid range of shortest sequence in batch
            batch = prepare_target_batch(cgyro, actual_num_queries, subkey, 
                                         max_query_idx=int(lengths.min()) if use_padded else None)
            
            # Shard batch across devices
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )
            
            # Train step
            state, loss, loss_recon = train_step(state, batch)
            step = int(state.step)
            
            # Logging
            if step % config.logging.log_interval == 0:
                loss_val = loss.item()
                loss_recon_val = loss_recon.item()
                current_lr = lr(step)
                
                if jax.process_index() == 0:
                    print(f"step: {step}, loss: {loss_val:.3e}, seq_len: {seq_len}, lr: {current_lr:.3e}")
                    
                    if use_wandb:
                        log_dict = {
                            "loss": loss_val,
                            "loss_recon": loss_recon_val,
                            "lr": current_lr,
                            "epoch": epoch,
                            "seq_len": seq_len,
                        }
                        wandb.log(log_dict, step=step)
            
            # Save checkpoint
            if step % config.saving.save_interval == 0:
                save_checkpoint(ckpt_mngr, state)
        
        end_time = time.time()
        if jax.process_index() == 0:
            print(f"Epoch {epoch} completed in {end_time - start_time:.2f}s")

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

