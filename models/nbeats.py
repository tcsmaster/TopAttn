# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
N-BEATS + TopAttn Model.
"""

from typing import Tuple
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = t.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / np.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = t.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    """
    Implementaiton of multi-headed attention.
    """

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        """
        input_dim: length of the time series window
        embed_dim: hidden diemension of the attention model
        num_heads: number of attention heads
        """
        super().__init__()
        assert input_dim % num_heads == 0, (
            "Embedding dimension must be divisible by num_heads"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(self.input_dim, 3 * self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: t.Tensor):
        batch_size, input_dim, _ = x.size()
        # Separate Q, K, V from linear output
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, self.input_dim, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, InputDim, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, InputDim, Head, Dims]
        values = values.reshape(batch_size, self.input_dim, self.embed_dim)
        o = self.o_proj(values)
        return o


class EncoderBlock(nn.Module):
    """
    Implementaiton of an Encoder Transformer block.
    """

    def __init__(self, input_dim, num_heads, dim_ff, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_ff - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiHeadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_ff),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ff, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.norm1(self.self_attn(x, mask=mask))
        x = x + self.dropout(attn_out)

        # MLP part
        linear_out = self.norm2(self.linear_net(x))
        x = x + self.dropout(linear_out)

        return x


class TopAttn(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, forecast_size):
        super().__init__()
        self.attn = MultiHeadAttention(input_dim, embed_dim, n_heads)
        self.final_layer = nn.Linear(
            in_features=2 * input_dim, out_features=forecast_size
        )

    def forward(self, x: t.Tensor):
        attn_output = self.attn(x)
        nbeats_input = t.concat((attn_output, x), dim=1)
        output = self.final_layer(nbeats_input)
        return output


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        input_size,
        theta_size: int,
        basis_function: nn.Module,
        layers: int,
        layer_size: int,
    ):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)]
            + [
                nn.Linear(in_features=layer_size, out_features=layer_size)
                for _ in range(layers - 1)
            ]
        )
        self.basis_parameters = nn.Linear(
            in_features=layer_size, out_features=theta_size
        )
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        self.polynomial_size = (
            degree_of_polynomial + 1
        )  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=np.float) / backcast_size, i
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )
        self.forecast_time = nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=np.float) / forecast_size, i
                        )[None, :]
                        for i in range(self.polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )

    def forward(self, theta: t.Tensor):
        backcast = t.einsum(
            "bp,pt->bt", theta[:, self.polynomial_size :], self.backcast_time
        )
        forecast = t.einsum(
            "bp,pt->bt", theta[:, : self.polynomial_size], self.forecast_time
        )
        return backcast, forecast


class SeasonalityBasis(nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32)
            / harmonics,
        )[None, :]
        backcast_grid = (
            -2
            * np.pi
            * (np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )
        forecast_grid = (
            2
            * np.pi
            * (np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size)
            * self.frequency
        )
        self.backcast_cos_template = nn.Parameter(
            t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.backcast_sin_template = nn.Parameter(
            t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.forecast_cos_template = nn.Parameter(
            t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
            requires_grad=False,
        )
        self.forecast_sin_template = nn.Parameter(
            t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
            requires_grad=False,
        )

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic : 3 * params_per_harmonic],
            self.backcast_cos_template,
        )
        backcast_harmonics_sin = t.einsum(
            "bp,pt->bt", theta[:, 3 * params_per_harmonic :], self.backcast_sin_template
        )
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum(
            "bp,pt->bt", theta[:, :params_per_harmonic], self.forecast_cos_template
        )
        forecast_harmonics_sin = t.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic : 2 * params_per_harmonic],
            self.forecast_sin_template,
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast
