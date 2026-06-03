import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from transformers import GPT2Config, GPT2Model



class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Remember the residual connection
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, D = x.size()
        device = x.device

        positions = torch.arange(T, device=device, dtype=torch.float).unsqueeze(1)  # (T,1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() *
                             (-math.log(10000.0) / self.d_model))  # (D//2,)

        pe = torch.zeros(T, self.d_model, device=device)  # (T, D)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)  # (1, T, D)

        return x + pe  # broadcast to (B, T, D)

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        vocab_size: int = None,
    ):
        super().__init__()

        config = GPT2Config(
            n_embd=embed_size,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=max_position_embeddings,
            vocab_size=vocab_size or 1,
            use_cache=False,
        )
        self.decoder = GPT2Model(config)

    def forward(self, input_ids):

        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=None,
        )
        # outputs.last_hidden_state.shape = (B, T, embed_size)
        return outputs.last_hidden_state


class OfficialGPT2Decoder(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 32,
        vocab_size: int = None,
        train = False
    ):
        super().__init__()
        config = GPT2Config(
            n_embd=512,
            n_layer=num_layers,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            n_positions=max_position_embeddings,
            vocab_size=vocab_size or 1,
            use_cache=False,
        )
        self.decoder = GPT2Model(config)
        if not train:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x, attention_mask=None, position_ids=None):

        outputs = self.decoder(
            inputs_embeds=x,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.last_hidden_state

    
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size
