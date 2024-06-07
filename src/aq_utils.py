import math, scipy
import torch
from typing import Tuple, NamedTuple


NF4_OFFSET = 0.9677083  # Magic number?


class QuantScheme(NamedTuple):
    values: torch.Tensor
    boundaries: torch.Tensor


def dimwise_absmax(A: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.max(
        torch.abs(A),
        dim=dim,
        keepdim=True).values


def blockwise_absmax(
    A: torch.Tensor,
    num_bits_0: int,  # of the second-level quantization
    num_bits_1: str,  # of the second-level quantization states
    block_size_0: int,
    block_size_1: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (TODO) Double check this
    if A.dtype != torch.float32:
        raise ValueError(f"Expected float32, but got {A.dtype}")
    if num_bits_1 == "bf16":
        dtype = torch.bfloat16
    elif num_bits_1 == "fp16":
        dtype = torch.float16
    elif num_bits_1 == "fp32":
        dtype = torch.float32
    else:
        raise ValueError

    # Compute the second-level quantization
    scales_0 = A.view(-1, block_size_1, block_size_0)
    scales_1 = dimwise_absmax(scales_0, dim=2)
    # Notice that we use the `.min` as the offset.
    # This guarantees that the smallest number after
    # quantization will be at least `offset_1`, which is
    # positive because `scales_1` is non-negative.
    offset_1 = scales_1.min()
    scales_2 = scales_1 - offset_1
    scales_3 = dimwise_absmax(scales_2, dim=1)
    # (TODO) Double check this
    scales_3 = (
        scales_3
        .to(dtype=dtype)
        .to(dtype=scales_3.dtype))

    # Reconstruct the first-level quantization scales
    scales_3_ = torch.broadcast_to(scales_3, scales_2.shape)
    # (Unsigned) int8 quantization of the first-level scales
    scales_2_ = scales_2
    #scales_2_ = quantize_with_scheme_2(
        #scales_2,
        #scales=scales_3_,
        #num_bits=num_bits_0,
        #dtype="uint")
    scales_1_ = scales_2_ + offset_1

    # `scales_q` is the `scales` for quantizing `A`
    # `scales_dq` is the `scales` for dequantizing `A`
    scales_q = torch.broadcast_to(scales_1, scales_0.shape)
    scales_dq = torch.broadcast_to(scales_1_, scales_0.shape)
    scales_q = scales_q.reshape(A.shape)
    scales_dq = scales_dq.reshape(A.shape)
    return scales_q, scales_dq

def create_normal_float_scheme(
    num_bits: int,
    device: torch.device,
) -> QuantScheme:
    # This is essentially what NF4 does.
    sigma = -1. / (
        math.sqrt(2) *
        scipy.special.erfinv(1 - 2 * NF4_OFFSET))
    qdist = torch.distributions.normal.Normal(
        loc=0.,
        scale=sigma)

    quantiles_left = torch.linspace(
        1. - NF4_OFFSET,
        0.5,
        2 ** (num_bits - 1))
    quantiles_right = torch.linspace(
        0.5,
        NF4_OFFSET,
        2 ** (num_bits - 1) + 1)
    # remove the duplicated `0.5`
    quantiles = torch.cat([
        quantiles_left[:-1],
        quantiles_right],
        dim=0)
    values = qdist.icdf(quantiles)
    return create_quantization_scheme(
        values=values,
        device=device)


def create_quantization_scheme(
    values: torch.Tensor,
    device: torch.device,
) -> QuantScheme:
    inf_tensor = torch.tensor([torch.inf])
    boundaries = (values[1:] + values[:-1]) / 2.
    boundaries = torch.cat([-inf_tensor, boundaries, inf_tensor], dim=0)

    values = values.to(device=device)
    boundaries = boundaries.to(device=device)
    if values.ndim != 1 or boundaries.ndim != 1:
        raise ValueError
    if values.shape[0] != boundaries.shape[0] - 1:
        raise ValueError
    return QuantScheme(
        values=values,
        boundaries=boundaries)

def quantize_with_scheme(
    A: torch.Tensor,
    qscheme: QuantScheme,
    scales_q: torch.Tensor,
    scales_dq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.shape != scales_q.shape:
        raise ValueError
    if A.shape != scales_dq.shape:
        raise ValueError
    A_scaled = A / scales_q
    # `-1` because this function assigns to the right bucket
    A_quantized = torch.bucketize(
        A_scaled,
        qscheme.boundaries,
        right=False) - 1
    A_dequantized = qscheme.values[A_quantized]
    A_dequantized = A_dequantized * scales_dq
    return A_quantized, A_dequantized

def svd_decomposition(
    A: torch.Tensor,
    randomized: bool,
    num_ranks: int,
    num_oversampling: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    if randomized is False:
        U, S, VT = torch.linalg.svd(A, full_matrices=False)
    elif randomized is True:
        U, S, V = torch.svd_lowrank(A, num_ranks + num_oversampling)
        # https://pytorch.org/docs/stable/_modules/torch/_lowrank.html#svd_lowrank
        VT = V.mH
    else:
        raise ValueError(f"`randomized` {randomized} not supported")

    S_sqrt = torch.sqrt(S)
    L1 = U * S_sqrt.unsqueeze(dim=0)
    L2 = VT * S_sqrt.unsqueeze(dim=1)
    L1k = L1[:, :num_ranks]
    L2k = L2[:num_ranks, :]
    return L1k, L2k
