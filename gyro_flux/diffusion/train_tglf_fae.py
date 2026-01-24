"""Training script for TGLF FAE.

Trains the TGLF conditioning encoder/decoder to reconstruct
TGLF spectra at arbitrary ky positions.

Usage:
    python train_tglf_fae.py --config=configs/tglf_fae.py
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
import numpy as np
from sklearn.model_selection import train_test_split

from gyro_flux.models import TGLFEncoder, TGLFDecoder
from gyro_flux.data_utils import TGLFDataset, create_tglf_dataloader, get_paired_folders
from gyro_flux.diffusion.model_utils import (
    create_tglf_train_step,
    prepare_tglf_batch,
    tglf_loss_fn,
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


def create_tglf_fae_state(config, encoder, decoder, tx):
    """Initialize encoder and decoder parameters."""
    # Dummy inputs for initialization
    x = jnp.ones(config.x_dim)
    ky_query = jnp.ones(config.ky_query_dim)

    encoder_params = encoder.init(random.PRNGKey(config.seed), x)
    z_c = encoder.apply(encoder_params, x)  # (B, z_c_dim)
    decoder_params = decoder.init(random.PRNGKey(config.seed), z_c, ky_query)
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
    """Main training loop."""
    
    # Override data path if provided via flag
    if FLAGS.data_path is not None:
        config.dataset.data_path = FLAGS.data_path
    
    if config.dataset.data_path is None:
        raise ValueError("data_path must be set in config or via --data_path flag")
    
    # Initialize models
    encoder = TGLFEncoder(**config.model.encoder)
    decoder = TGLFDecoder(**config.model.decoder)
    
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_tglf_fae_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create train step function
    train_step = create_tglf_train_step(encoder, decoder, mesh)

    # Create dataset split (80/10/10: train/val/test)
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

    # Create train dataset
    train_dataset = TGLFDataset(
        data_path=config.dataset.data_path,
        paired_only=True,
        normalize=False,
        folder_list=train_folders,
    )

    # Create val dataset
    val_dataset = TGLFDataset(
        data_path=config.dataset.data_path,
        paired_only=True,
        normalize=False,
        folder_list=val_folders,
    )

    train_loader = create_tglf_dataloader(
        train_dataset,
        batch_size=config.dataset.train_batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )

    val_loader = create_tglf_dataloader(
        val_dataset,
        batch_size=config.dataset.test_batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=False,
        drop_last=False,
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
    print(f"Batch size: {config.dataset.train_batch_size * num_devices}")
    
    epoch = 0
    train_loss_val = 0.0  # Track last training loss for epoch summary
    while step < config.training.max_steps:
        epoch += 1
        start_time = time.time()

        for batch_data in train_loader:
            if step >= config.training.max_steps:
                break

            rng_key, subkey = jax.random.split(rng_key)

            # Convert to JAX arrays
            tglf = jnp.array(batch_data)  # (B, 21, 108)

            # Prepare batch with random query sampling
            batch = prepare_tglf_batch(tglf, num_queries, subkey)

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
                loss_val = train_loss_val
                loss_recon_val = loss_recon.item()
                current_lr = lr(step)
                
                if jax.process_index() == 0:
                    print(f"step: {step}, loss: {loss_val:.3e}, lr: {current_lr:.3e}")
                    
                    if use_wandb:
                        log_dict = {
                            "loss/train": loss_val,
                            "loss_recon/train": loss_recon_val,
                            "lr": current_lr,
                            "epoch": epoch,
                        }
                        wandb.log(log_dict, step=step)
            
            # Save checkpoint
            if step % config.saving.save_interval == 0:
                save_checkpoint(ckpt_mngr, state)

        # Validation at end of each epoch
        if jax.process_index() == 0:
            print(f"Epoch {epoch} completed, running validation...")

        val_loss_sum = 0.0
        val_loss_recon_sum = 0.0
        val_batches = 0
        val_rng_key = jax.random.PRNGKey(config.seed + epoch)

        for val_batch_data in val_loader:
            val_rng_key, val_subkey = jax.random.split(val_rng_key)

            tglf_val = jnp.array(val_batch_data)  # (B, 21, 108)

            # Prepare batch with random query sampling
            val_batch = prepare_tglf_batch(tglf_val, num_queries, val_subkey)

            # Shard batch across devices
            val_batch = multihost_utils.host_local_array_to_global_array(
                val_batch, mesh, P("batch")
            )

            # Compute validation loss (no gradients)
            val_loss, val_metrics = tglf_loss_fn(
                encoder, decoder, state.params, val_batch
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
        from gyro_flux.diffusion.configs.tglf_fae import get_config
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

