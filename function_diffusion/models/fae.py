from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, normal, xavier_uniform

import flax.linen as nn
from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size[0], dtype=jnp.float32)
    grid_w = jnp.arange(grid_size[1], dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h, indexing="ij")  # here w goes first
    grid = jnp.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return jnp.expand_dims(pos_embed, 0)


class PatchEmbed(nn.Module):
    patch_size: tuple = (16, 16)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        b, h, w, c = inputs.shape
        x = nn.Conv(
            self.emb_dim,
            (self.patch_size[0], self.patch_size[1]),
            (self.patch_size[0], self.patch_size[1]),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (b, -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x


class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


# =============================================================================
# Skip-Perceiver Components (Masked Attention for Variable-Length Sequences)
# =============================================================================

def create_attention_mask(lengths: jnp.ndarray, max_len: int) -> jnp.ndarray:
    """Create bidirectional attention mask from sequence lengths.

    Args:
        lengths: (B,) actual sequence lengths per sample
        max_len: T_pad, the padded sequence length

    Returns:
        mask: (B, 1, T, T) attention mask where:
              True = attend, False = don't attend (will become -inf)
    """
    # Step 1: Create 1D validity mask (B, T)
    # Example: lengths=[3,2], max_len=4 → [[T,T,T,F], [T,T,F,F]]
    positions = jnp.arange(max_len)  # [0, 1, 2, ..., T-1]
    mask_1d = positions[None, :] < lengths[:, None]  # (B, T)

    # Step 2: Create 2D mask (B, T, T)
    # Position i can attend to position j iff BOTH are valid
    # mask_2d[b, i, j] = mask_1d[b, i] AND mask_1d[b, j]
    mask_2d = mask_1d[:, :, None] & mask_1d[:, None, :]  # (B, T, T)

    # Step 3: Add head dimension (B, 1, T, T) for broadcasting over heads
    mask_2d = mask_2d[:, None, :, :]

    return mask_2d


def create_cross_attention_mask(q_lengths: jnp.ndarray, kv_lengths: jnp.ndarray,
                                 q_max_len: int, kv_max_len: int) -> jnp.ndarray:
    """Create cross-attention mask from query and key-value lengths.

    Args:
        q_lengths: (B,) actual query lengths per sample
        kv_lengths: (B,) actual key-value lengths per sample
        q_max_len: Padded query length
        kv_max_len: Padded key-value length

    Returns:
        mask: (B, 1, Q, KV) attention mask
    """
    # Query validity: (B, Q)
    q_positions = jnp.arange(q_max_len)
    q_valid = q_positions[None, :] < q_lengths[:, None]

    # KV validity: (B, KV)
    kv_positions = jnp.arange(kv_max_len)
    kv_valid = kv_positions[None, :] < kv_lengths[:, None]

    # Cross mask: (B, Q, KV) - query i can attend to kv j iff both valid
    mask_2d = q_valid[:, :, None] & kv_valid[:, None, :]

    # Add head dimension
    return mask_2d[:, None, :, :]


class MaskedSelfAttnBlock(nn.Module):
    """Self-attention with explicit padding mask support.

    Unlike SelfAttnBlock, this explicitly masks padded positions with -inf
    in attention scores before softmax, ensuring zero attention to padding.
    """
    num_heads: int
    emb_dim: int
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mask=None):
        """
        Args:
            x: (B, T, D) input sequence
            mask: (B, 1, T, T) attention mask (True=attend, False=mask out)
                  If None, no masking is applied.

        Returns:
            (B, T, D) output with masked attention
        """
        B, T, D = x.shape
        head_dim = D // self.num_heads

        # Pre-norm
        x_norm = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        # Project to Q, K, V
        qkv = nn.Dense(3 * D, name='qkv')(x_norm)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head: (B, T, D) → (B, heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores: (B, heads, T, T)
        scale = head_dim ** -0.5
        scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale

        # EXPLICIT MASKING: Set padding positions to -inf
        if mask is not None:
            # mask: (B, 1, T, T), broadcasts over heads
            # Where mask is False, set score to large negative value
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        # Softmax → attention weights (softmax(-inf) = 0)
        attn = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)

        # Reshape back: (B, heads, T, head_dim) → (B, T, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Output projection
        out = nn.Dense(D, name='out_proj')(out)

        # Residual connection
        x = x + out

        # MLP block with pre-norm
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class MaskedCrossAttnBlock(nn.Module):
    """Cross-attention with explicit key-value mask support.

    Queries attend to key-value pairs, with explicit masking for padded KV positions.
    """
    num_heads: int
    emb_dim: int
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs, kv_mask=None):
        """
        Args:
            q_inputs: (B, Q, D) query sequence
            kv_inputs: (B, KV, D) key-value sequence
            kv_mask: (B, KV) or (B, 1, 1, KV) mask for KV positions
                     True=valid, False=padding. If None, no masking.

        Returns:
            (B, Q, D) output
        """
        B, Q, D = q_inputs.shape
        KV = kv_inputs.shape[1]
        head_dim = D // self.num_heads

        # Pre-norm
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        # Project Q from queries, K and V from kv_inputs
        q_proj = nn.Dense(D, name='q_proj')(q)
        k_proj = nn.Dense(D, name='k_proj')(kv)
        v_proj = nn.Dense(D, name='v_proj')(kv)

        # Reshape for multi-head
        q_proj = q_proj.reshape(B, Q, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k_proj = k_proj.reshape(B, KV, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v_proj = v_proj.reshape(B, KV, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention scores: (B, heads, Q, KV)
        scale = head_dim ** -0.5
        scores = jnp.einsum('bhid,bhjd->bhij', q_proj, k_proj) * scale

        # Apply KV mask if provided
        if kv_mask is not None:
            # Ensure mask has shape (B, 1, 1, KV) for broadcasting
            if kv_mask.ndim == 2:  # (B, KV)
                kv_mask = kv_mask[:, None, None, :]
            elif kv_mask.ndim == 3:  # (B, 1, KV)
                kv_mask = kv_mask[:, :, None, :]
            # Where mask is False, set to -inf
            scores = jnp.where(kv_mask, scores, jnp.finfo(scores.dtype).min)

        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bhij,bhjd->bhid', attn, v_proj)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(B, Q, D)
        out = nn.Dense(D, name='out_proj')(out)

        # Residual
        x = q_inputs + out

        # MLP with pre-norm
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation for conditioning.

    Applies scale (gamma) and shift (beta) to input features based on
    conditioning vector. Used for physics conditioning in Skip-Perceiver.

    Reference: Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
    """
    emb_dim: int

    @nn.compact
    def __call__(self, x, conditioning):
        """
        Args:
            x: (B, T, D) input features
            conditioning: (B, C) conditioning vector (e.g., physics params)

        Returns:
            (B, T, D) modulated features
        """
        # Project conditioning to scale and shift
        gamma = nn.Dense(self.emb_dim, name='gamma')(conditioning)  # (B, D)
        beta = nn.Dense(self.emb_dim, name='beta')(conditioning)    # (B, D)

        # Center gamma around 1 (so default is identity transform)
        gamma = 1.0 + gamma

        # Apply FiLM: x' = gamma * x + beta
        # gamma/beta are (B, D), broadcast over T dimension
        return gamma[:, None, :] * x + beta[:, None, :]


class ReadoutPooling(nn.Module):
    """Pool variable-length sequence to fixed-size representation.

    Uses learnable readout tokens that aggregate sequence information via
    cross-attention. Similar to Perceiver, but applied AFTER full self-attention
    (not before), preserving more information.
    """
    num_readout: int = 64
    emb_dim: int = 128
    num_heads: int = 8
    depth: int = 2
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mask=None):
        """
        Args:
            x: (B, T, D) full sequence from encoder
            mask: (B, T) validity mask (True=valid, False=padding)

        Returns:
            (B, num_readout, D) fixed-size representation for decoder
        """
        B = x.shape[0]

        # Learnable readout queries
        readout = self.param('readout',
                            normal(stddev=0.02),
                            (self.num_readout, self.emb_dim))
        # Broadcast to batch
        readout = jnp.broadcast_to(readout, (B, self.num_readout, self.emb_dim))
        # Make it a proper array (not a broadcast view) for gradient flow
        readout = jnp.array(readout)

        # Cross-attention: readout queries, sequence is key/value
        # This aggregates information from the full sequence into fixed tokens
        for _ in range(self.depth):
            readout = MaskedCrossAttnBlock(
                num_heads=self.num_heads,
                emb_dim=self.emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps
            )(readout, x, kv_mask=mask)

        # Final layer norm
        readout = nn.LayerNorm(epsilon=self.layer_norm_eps)(readout)

        return readout  # (B, num_readout, D)


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, L,  D) --> (B, L', D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (L', D)
                             )

        latents = repeat(latents, 'l d -> b l d', b=x.shape[0])  # (B, L', D)
        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)

        latents = nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)
        return latents


class Encoder(nn.Module):
    patch_size: int
    grid_size: Tuple
    emb_dim: int
    num_latents: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5
    pos_emb_init: Callable = get_2d_sincos_pos_embed

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape

        # Patch embedding
        x = PatchEmbed(self.patch_size, self.emb_dim)(x)

        pos_emb = self.variable(
            "pos_emb",
            "enc_pos_emb",
            self.pos_emb_init,
            self.emb_dim,
            (self.grid_size[0] // self.patch_size[0], self.grid_size[1] // self.patch_size[1]),
        )

        # Interpolate positional embeddings to match the input shape
        pos_emb_interp = pos_emb.value.reshape(1,
                                             self.grid_size[0] // self.patch_size[0],
                                             self.grid_size[1] // self.patch_size[1],
                                             self.emb_dim)
        pos_emb_interp = jax.image.resize(pos_emb_interp,
                                            (1, h // self.patch_size[0], w // self.patch_size[1], self.emb_dim),
                                            method='bilinear')
        pos_emb_interp = rearrange(pos_emb_interp, 'b h w d -> b (h w) d')
        x = x + pos_emb_interp

        # Embed into tokens of the same length as the latents
        x = PerceiverBlock(emb_dim=self.emb_dim, depth=2, num_heads=self.num_heads, num_latents=self.num_latents)(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        # Transformer
        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Periods for different axes
    axis: Tuple[int]  # Axes where the period embeddings are to be applied

    def setup(self):
        # Initialize period parameters and store them in a flax frozen dict
        self.period_params = {f"period_{idx}": period for idx, period in enumerate(self.period)}

    @nn.compact
    def __call__(self, x):
        """
        Apply the period embeddings to the specified axes.
        """
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)

        return jnp.hstack(y)


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    period: Union[None, Dict] = None
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, coords):
        b, n, c = x.shape

        # # Embed periodic boundary conditions if specified
        if self.period is True:
            # Hardcode the periodicity, assuming the domain is [0, 1]x[0, 1]
            coords = PeriodEmbs(period=(2 * jnp.pi, 2 * jnp.pi), axis=(0, 1))(coords)

        coords = FourierEmbs(embed_scale=self.fourier_freq, embed_dim=self.dec_emb_dim)(coords)
        coords = repeat(coords, 'd -> b n d', n=1, b=b)

        x = nn.Dense(self.dec_emb_dim)(x)
        for _ in range(self.dec_depth):
            coords = CrossAttnBlock(num_heads=self.dec_num_heads,
                               emb_dim=self.dec_emb_dim,
                               mlp_ratio=self.mlp_ratio,
                               layer_norm_eps=self.layer_norm_eps)(coords, x)

        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords)
        # x = nn.Dense(self.out_dim)(x)

        x = Mlp(num_layers=self.num_mlp_layers,
                hidden_dim=self.dec_emb_dim,
                out_dim=self.out_dim,
                layer_norm_eps=self.layer_norm_eps)(x)

        return x