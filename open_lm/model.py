# Modified from https://github.com/facebookresearch/llama

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import json
import re
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import xformers.ops as xops
from xformers.components.positional_embedding import RotaryEmbedding


_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    seq_len: int = 2048
    pre_ln: bool = False

    pos_embed_type: str = 'rope'
    weight_tying: bool = False
    attn_type: str = 'xformers'


class IdentityEmbedding(nn.Module):
    def forward(q, k):
        return q, k

# TODO: follow up on why this is necessary
# https://github.com/facebookresearch/xformers/issues/742
class RotaryWithCast(RotaryEmbedding):
    def forward(self, q, k, v):
        q, k = super().forward(q, k)
        return q.to(v.dtype), k.to(v.dtype), v

def xformers_attn(xq, xk, xv, is_causal):
    # NOTE: in another project noticed F.scaled_dot_product gave different
    # results than nn.MultiheadAttention. That is why we have xformers option.
    # NOTE: Why is this not tri-u? They did this in the xformers example, but need to Investigate.
    # NOTE: We are casting xq and xk to bfloats -- need to experiment with instead casting xv to float.
    # NOTE: Should we not initialize the mask before return statement?
    mask = None
    if is_causal:
        mask = xops.LowerTriangularMask()
    return xops.memory_efficient_attention(
        xq, xk, xv, attn_bias=mask
    )


class CustomAttn(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.in_proj = nn.Linear(
            args.dim,
            3 * args.n_heads * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )
        self.rope_cache = None
        if args.pos_embed_type == 'rope':
            self.pos_embed = RotaryWithCast(self.head_dim, args.seq_len)
        else:
            # TODO: add other positional embeddings such as Alibi.
            self.pos_embed = IdentityEmbedding()

        if args.attn_type == 'xformers':
            self.attn_fn = xformers_attn
        else:
            self.attn_fn = F.scaled_dot_product_attention

        # initialize weights.
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.in_proj.weight, std=std, a=-3 * std, b=3 * std)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(self.out_proj.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor, is_causal: bool = False):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.in_proj(x).chunk(3, dim=-1)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk, xv = self.pos_embed(xq, xk, xv)

        # NOTE: when using rotary embeddings the output is float32, so we cast it back.
        # Look into the effect of this.

        output = self.attn_fn(
            xq.to(xv.dtype), 
            xk.to(xv.dtype), 
            xv, 
            is_causal=is_causal
        )

        output = output.view(bsz, seqlen, -1)

        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attn_type = args.attn_type
        if self.attn_type == 'xformers' or args.pos_embed_type == 'rope':
            self.attention = CustomAttn(layer_id, args)
        else:
            self.attention = nn.MultiheadAttention(args.dim, args.n_heads, bias=False, batch_first=True)

        # TODO: add other options for feed forward beyond 2/3 and 4 SwiGLU.
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.feed_forward = xops.SwiGLU(args.dim, hidden_dim, args.dim, bias=False)
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.LayerNorm(args.dim, eps=args.norm_eps)
        self.attention.seq_len = args.seq_len

        # initialize weights.
        std = 1.0 / math.sqrt(args.dim)
        torch.nn.init.trunc_normal_(self.feed_forward.w12.weight, std=std, a=-3 * std, b=3 * std)
        std = 1.0 / math.sqrt(hidden_dim)
        std = std / math.sqrt(2 * (layer_id + 1))
        torch.nn.init.trunc_normal_(self.feed_forward.w3.weight, std=std, a=-3 * std, b=3 * std)

    def attention_fwd(self, x: torch.Tensor, is_causal: bool = False):
        if self.attn_type == 'xformers':
            return self.attention(x, is_causal=is_causal)
        else:
            return self.attention(x, x, x, need_weights=False, is_causal=is_causal)[0]

    def forward(self, x: torch.Tensor):
        h = x + self.attention_fwd(self.attention_norm(x), is_causal=True)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, embed=True):
        super().__init__()
        # TODO: the texttransformer in openclip has a cast_dtype, look into why and if necessary here.
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.seq_len = params.seq_len
        self.pre_ln = params.pre_ln
        self.embed = embed
        self.pos_embed_type = params.pos_embed_type
        self.weight_tying = params.weight_tying

        if embed:
            self.tok_embeddings = nn.Embedding(
                params.vocab_size, params.dim
            )

        if self.pos_embed_type == 'learned':
            self.pos_embeddings = nn.Parameter(torch.empty(params.seq_len, params.dim))
            torch.nn.init.normal_(self.pos_embeddings, std=0.01)

        # This is False in llama, but True in OpenCLIP/CLIP.
        # Also called Stable Embedding (see Tim's 8bit Adam paper)
        if self.pre_ln:
            self.pre_norm = nn.LayerNorm(params.dim, eps=params.norm_eps)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = nn.LayerNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False
        )
        if self.weight_tying:
            self.tok_embeddings.weight = self.output.weight

        self.grad_checkpointing = False

        # initialize weights.
        std = 1.0 / math.sqrt(params.dim)
        torch.nn.init.trunc_normal_(self.output.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.tok_embeddings.weight, std=std, a=-3 * std, b=3 * std)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def forward(self, tokens: torch.Tensor, output_hidden_states: bool = False):
        h = self.tok_embeddings(tokens) if self.embed else tokens

        if self.pos_embed_type == 'learned':
            h = h + self.pos_embeddings[None, :h.size(1), :]

        if self.pre_ln:
            h = self.pre_norm(h)
            
        hidden_states = []
        for layer in self.layers:
            if output_hidden_states:
                hidden_states.append(h)
            if self.grad_checkpointing:
                h = checkpoint(layer, h)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h)

        if output_hidden_states:
            hidden_states.append(h)

        # llama casts to float here, not sure why.
        if output_hidden_states:
            return output.float(), [h.float() for h in hidden_states]
        return output.float()


def create_model(args):

    cfg =  deepcopy(_MODEL_CONFIGS[args.model])

    model_args = ModelArgs(
        dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        seq_len=cfg['seq_len'],
        vocab_size=cfg['vocab_size'],
        pre_ln=cfg['pre_ln'],
        pos_embed_type=cfg['pos_embed_type'],
        weight_tying=cfg['weight_tying'],
        attn_type=cfg['attn_type'],
    )
    model = Transformer(model_args)

    return model
