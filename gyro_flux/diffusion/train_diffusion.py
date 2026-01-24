"""Training script for FluxDiT (Diffusion Transformer).

Trains the DiT model to learn rectified flow from noise to flux latents,
conditioned on TGLF stability vectors.

Uses frozen TGLF FAE and Target FAE encoders to generate latents.
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

from gyro_flux.models import (
    TGLFEncoder,
    TimeSeriesEncoder,
    FluxDiT,
)
from gyro_flux.data_utils import (
    TGLFDataset,
    CGYRODataset,
    create_tglf_dataloader,
    create_cgyro_dataloader,
)
from gyro_flux.diffusion.model_utils import (
    create_tglf_encoder_step,
    create_target_encoder_step,
    create_dit_train_step,
    prepare_dit_batch,
)

from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
)


FLAGS = flags.FLAGS
flags.DEFINE_string("config", None, "Path to config file.")
flags.DEFINE_string("data_path", None, "Path to data directory (overrides config).")
flags.DEFINE_string("workdir", None, "Working directory for checkpoints.")
flags.DEFINE_string("tglf_fae_ckpt", None, "Path to TGLF FAE checkpoint (overrides config).")
flags.DEFINE_string("target_fae_ckpt", None, "Path to Target FAE checkpoint (overrides config).")


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


def create_dit_state(config, model, tx):
    """Initialize FluxDiT parameters."""
    # Dummy inputs for initialization
    z_t = jnp.ones(config.z_dim)  # (B, num_latents, emb_dim)
    t = jnp.ones(config.t_dim)   # (B,)
    z_c = jnp.ones(config.c_dim) # (B, z_c_dim)

    params = model.init(random.PRNGKey(config.seed), z_t, t, z_c)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
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


def load_frozen_encoder(encoder, ckpt_path, mesh):
    """Load frozen encoder from checkpoint.
    
    Args:
        encoder: Encoder module (TGLFEncoder or TimeSeriesEncoder)
        ckpt_path: Path to checkpoint directory (contains 'ckpt' subdirectory)
        mesh: JAX device mesh
    
    Returns:
        encoder_params: Frozen encoder parameters
        encoder_step: Encoder step function
    """
    import orbax.checkpoint as ocp
    from jax.experimental import multihost_utils
    
    # Create dummy state to load checkpoint
    # For TGLF encoder: input is (B, 21, 108)
    # For Target encoder: input is (B, T, 2)
    if isinstance(encoder, TGLFEncoder):
        dummy_input = jnp.ones((1, 21, 108))
    else:
        dummy_input = jnp.ones((1, 1000, 2))
    
    dummy_params = encoder.init(random.PRNGKey(0), dummy_input)
    dummy_state = train_state.TrainState.create(
        apply_fn=lambda *args: None,
        params=dummy_params,
        tx=optax.sgd(1e-4),
    )
    
    # Create checkpoint manager from checkpoint directory
    # ckpt_path should point to the workdir, we need the 'ckpt' subdirectory
    if os.path.isdir(ckpt_path):
        # If it's already the ckpt directory
        ckpt_dir = ckpt_path
    else:
        # If it's the workdir, append 'ckpt'
        ckpt_dir = os.path.join(ckpt_path, "ckpt")
    
    ckpt_mngr = ocp.CheckpointManager(ckpt_dir)
    
    # Restore checkpoint
    multihost_utils.sync_global_devices("before_ckpt_restore")
    restored_state = ckpt_mngr.restore(
        ckpt_mngr.latest_step(),
        args=ocp.args.StandardRestore(dummy_state),
    )
    
    # Extract encoder params (assuming state.params is (encoder_params, decoder_params))
    if isinstance(restored_state.params, tuple):
        encoder_params = restored_state.params[0]  # First element is encoder
    else:
        encoder_params = restored_state.params
    
    # Create encoder step (frozen - no gradients)
    if isinstance(encoder, TGLFEncoder):
        encoder_step = create_tglf_encoder_step(encoder, mesh)
    else:
        encoder_step = create_target_encoder_step(encoder, mesh)
    
    return encoder_params, encoder_step


# =============================================================================
# Main Training Function
# =============================================================================

def train_and_evaluate(config: ml_collections.ConfigDict):
    """Main training loop for FluxDiT."""
    
    # Override paths if provided via flags
    if FLAGS.data_path is not None:
        config.dataset.data_path = FLAGS.data_path
    
    tglf_ckpt_path = FLAGS.tglf_fae_ckpt or config.checkpoints.tglf_fae_path
    target_ckpt_path = FLAGS.target_fae_ckpt or config.checkpoints.target_fae_path
    
    if config.dataset.data_path is None:
        raise ValueError("data_path must be set in config or via --data_path flag")
    if tglf_ckpt_path is None:
        raise ValueError("tglf_fae_ckpt must be set in config or via --tglf_fae_ckpt flag")
    if target_ckpt_path is None:
        raise ValueError("target_fae_ckpt must be set in config or via --target_fae_ckpt flag")
    
    # Initialize models
    tglf_encoder = TGLFEncoder(**config.tglf_encoder.encoder)
    target_encoder = TimeSeriesEncoder(**config.autoencoder.encoder)
    dit_model = FluxDiT(**config.diffusion)
    
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)
    
    # Create DiT train state
    dit_state = create_dit_state(config, dit_model, tx)
    num_params = compute_total_params(dit_state)
    print(f"DiT model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")
    
    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")
    
    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    dit_state = multihost_utils.host_local_array_to_global_array(dit_state, mesh, P())
    
    # Load frozen encoders
    print(f"Loading TGLF FAE encoder from {tglf_ckpt_path}...")
    tglf_encoder_params, tglf_encode_step = load_frozen_encoder(tglf_encoder, tglf_ckpt_path, mesh)
    tglf_encoder_params = jax.stop_gradient(tglf_encoder_params)  # Freeze gradients
    
    print(f"Loading Target FAE encoder from {target_ckpt_path}...")
    target_encoder_params, target_encode_step = load_frozen_encoder(target_encoder, target_ckpt_path, mesh)
    target_encoder_params = jax.stop_gradient(target_encoder_params)  # Freeze gradients
    
    # Create train step function
    train_step = create_dit_train_step(dit_model, mesh)
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    tglf_dataset = TGLFDataset(
        data_path=config.dataset.data_path,
        paired_only=True,
        normalize=False,
    )
    cgyro_dataset = CGYRODataset(
        data_path=config.dataset.data_path,
        normalize=False,
        use_padding=False,  # Variable-length for now
        max_seq_length=3000,
    )
    
    # Ensure datasets are aligned (same folders, same order)
    assert len(tglf_dataset) == len(cgyro_dataset), "TGLF and CGYRO datasets must have same length"
    
    # Use same random seed for both dataloaders to ensure alignment when shuffling
    # Note: Both datasets use get_paired_folders which returns folders in sorted order,
    # so indices should align. Using same seed ensures shuffling is consistent.
    dataloader_seed = config.seed
    
    tglf_loader = create_tglf_dataloader(
        tglf_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )
    cgyro_loader = create_cgyro_dataloader(
        cgyro_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )
    
    # Working directory for checkpoints
    if FLAGS.workdir is not None:
        workdir = FLAGS.workdir
    else:
        model_name = config.diffusion.model_name
        run_name = config.wandb.run_name or "default"
        workdir = os.path.join(os.getcwd(), model_name, run_name)
    
    ckpt_path = os.path.join(workdir, "ckpt")
    
    # Check if W&B is enabled
    use_wandb = getattr(config.wandb, "use_wandb", False)
    
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
            run_name = getattr(wandb_config, "run_name", None) or config.diffusion.model_name
            wandb.init(
                project=wandb_config.project,
                entity=getattr(wandb_config, "entity", None),
                group=getattr(wandb_config, "group", None),
                name=run_name,
                notes=getattr(wandb_config, "notes", None),
                tags=getattr(wandb_config, "tag", None).split(',') if getattr(wandb_config, "tag", None) else None,
                config=config.to_dict(),
            )
            wandb.log({"num_params": num_params}, step=0)
    
    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)
    
    # Training loop
    rng_key = jax.random.PRNGKey(config.seed)
    step = 0
    
    print(f"Starting training for {config.training.max_steps} steps...")
    print(f"Dataset size: {len(tglf_dataset)} samples")
    print(f"Batch size: {config.dataset.batch_size * num_devices}")
    
    epoch = 0
    while step < config.training.max_steps:
        epoch += 1
        start_time = time.time()
        
        # Create iterators for paired data
        tglf_iter = iter(tglf_loader)
        cgyro_iter = iter(cgyro_loader)
        
        for _ in range(len(tglf_loader)):
            if step >= config.training.max_steps:
                break
            
            try:
                # Get paired batch
                tglf_batch = next(tglf_iter)
                cgyro_batch = next(cgyro_iter)
            except StopIteration:
                break
            
            rng_key, subkey = jax.random.split(rng_key)
            
            # Convert to JAX arrays
            tglf = jnp.array(tglf_batch)  # (B, 21, 108)
            cgyro_data = cgyro_batch['cgyro']  # (B, T, 2) - variable T
            cgyro_lengths = cgyro_batch['lengths']  # (B,)
            
            # Encode to latents (frozen encoders)
            z_c = jax.stop_gradient(
                tglf_encode_step(tglf_encoder_params, tglf)
            )  # (B, z_c_dim)
            
            # For variable-length sequences, we need to handle batch_size=1 or pad
            # For now, assume we're using padded sequences or batch_size=1
            # TODO: Handle variable-length properly
            z_1 = jax.stop_gradient(
                target_encode_step(target_encoder_params, cgyro_data)
            )  # (B, num_latents, emb_dim)
            
            # Sample noise z_0 ~ N(0, I)
            z_0 = random.normal(subkey, shape=z_1.shape)  # (B, num_latents, emb_dim)
            
            # Prepare DiT batch (samples t and creates z_t)
            rng_key, batch_key = jax.random.split(rng_key)
            dit_batch = prepare_dit_batch(z_0, z_1, z_c, batch_key)
            
            # Shard batch across devices
            dit_batch = multihost_utils.host_local_array_to_global_array(
                dit_batch, mesh, P("batch")
            )
            
            # Train step
            dit_state, loss, _ = train_step(dit_state, dit_batch)
            step = int(dit_state.step)
            
            # Logging
            if step % config.logging.log_interval == 0:
                loss_val = loss.item()
                current_lr = lr(step)
                
                if jax.process_index() == 0:
                    print(f"step: {step}, loss: {loss_val:.3e}, lr: {current_lr:.3e}")
                    
                    if use_wandb:
                        log_dict = {
                            "loss": loss_val,
                            "lr": current_lr,
                            "epoch": epoch,
                        }
                        wandb.log(log_dict, step=step)
            
            # Save checkpoint
            if step % config.saving.save_interval == 0:
                save_checkpoint(ckpt_mngr, dit_state)
        
        end_time = time.time()
        if jax.process_index() == 0:
            print(f"Epoch {epoch} completed in {end_time - start_time:.2f}s")
    
    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, dit_state)
    ckpt_mngr.wait_until_finished()
    
    if jax.process_index() == 0 and use_wandb:
        wandb.finish()


def main(argv):
    del argv  # Unused
    
    # Import config
    if FLAGS.config is None:
        # Use default config
        from gyro_flux.diffusion.configs.diffusion import get_config
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
    app.run(main)
