import math
import enum
import click
import torch
from dataclasses import dataclass
from typing import Tuple, Dict
from typing_extensions import Self
from pytorch_quantization import tensor_quant
from scipy.cluster.vq import kmeans2
import numpy as np
from tqdm import tqdm
import time

from models import misc_utils
from models import packbits_utils
from models import tensor_container_utils
from models.quantization_utils import (
    QuantConfig,
    QuantScheme,
    dimwise_absmax,
    blockwise_absmax,
    create_normal_float_scheme)


@dataclass
class PackedQuantizedTensorContainer(tensor_container_utils.TensorContainer):
    ptensor: packbits_utils.PackedBinaryTensorType
    padding_length: int
    shape: Tuple[int, ...]
    num_bits: int
    dtype: torch.dtype

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        num_bits: int,
    ) -> Self:
        # We only pack `uint8` tensors
        if tensor.dtype in [torch.int32, torch.int64, torch.float32]:
            uint8_tensor = tensor.to(dtype=torch.uint8)
            if (uint8_tensor != tensor).any():
                raise ValueError
        else:
            raise TypeError

        # Do the packing
        ptensor, padding_length = packbits_utils.pack_integer_tensors(
            tensor=uint8_tensor,
            num_bits=num_bits)

        return cls(
            ptensor=ptensor,
            padding_length=padding_length,
            shape=tensor.shape,  # same as `uint8_tensor.shape`
            num_bits=num_bits,
            dtype=tensor.dtype)  # we need the original dtype

    def to_tensor(self) -> torch.Tensor:
        uint8_tensor = packbits_utils.unpack_integer_tensors(
            packed_tensor=self.ptensor,
            padding_length=self.padding_length,
            num_bits=self.num_bits,
            shape=self.shape)
        return uint8_tensor.to(dtype=self.dtype)


@dataclass
class FloatTensorContainer(tensor_container_utils.TensorContainer):
    qtensor: packbits_utils.FloatTensorType
    qdtype: torch.dtype
    dtype: torch.dtype
    num_bits: str

    @staticmethod
    def get_qdtype(num_bits: str) -> torch.dtype:
        if num_bits == "bf16":
            qdtype = torch.bfloat16
        elif num_bits == "fp16":
            qdtype = torch.float16
        elif num_bits == "fp32":
            qdtype = torch.float32
        else:
            raise ValueError
        return qdtype

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        num_bits: str,
    ) -> Self:
        dtype = tensor.dtype
        qdtype = cls.get_qdtype(num_bits=num_bits)
        # (TODO) Double check this
        qtensor = tensor.to(dtype=qdtype)
        return cls(
            qtensor=qtensor,
            qdtype=qdtype,
            dtype=dtype,
            num_bits=num_bits)

    def to_tensor(self) -> torch.Tensor:
        return self.qtensor.to(dtype=self.dtype)


@dataclass
class UIntQuantizedTensorContainer(tensor_container_utils.TensorContainer):
    ctensor: PackedQuantizedTensorContainer
    cscales: FloatTensorContainer
    num_bits: int

    @staticmethod
    def preprocess_cscales(
        cscales: FloatTensorContainer,
        shape: torch.Size,
    ) -> torch.Tensor:
        scales = cscales.to_tensor()
        return torch.broadcast_to(scales, shape)

    @staticmethod
    def compute_qscales(scales: torch.Tensor, num_bits: int, unsigned: bool, assert_values: bool) -> torch.Tensor:
        # Computation must be in FP32 to prevent potential over flow.
        scales = scales.to(dtype=torch.float32)

        # This will cause graph breaks with `torch.compile`
        if assert_values is True:
            # `tensor_quant` has special handling of the following cases
            # but this should not happen in our case.
            epsilon = 1. / (1<<24)
            min_amax = scales.min()
            if min_amax < 0:
                raise ValueError
            if min_amax <= epsilon:
                raise ValueError

        max_bound = torch.tensor(
            (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0,
            device=scales.device)
        return max_bound / scales

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        cscales: FloatTensorContainer,
        num_bits: int,
    ) -> Self:
        scales = cls.preprocess_cscales(
            cscales=cscales,
            shape=tensor.shape)

        qtensor, _qscales = tensor_quant.tensor_quant(
            tensor,
            scales,
            num_bits,
            True)  # unsigned

        # Sanity checks
        qscales = cls.compute_qscales(
            scales=scales,
            num_bits=num_bits,
            unsigned=True,
            assert_values=True)

        if tensor.shape != scales.shape:
            raise ValueError
        if (qscales != _qscales).any():
            raise ValueError

        # Pack the quantized tensor
        ctensor = PackedQuantizedTensorContainer.from_tensor(
            tensor=qtensor,
            num_bits=num_bits)
        return cls(
            ctensor=ctensor,
            cscales=cscales,
            num_bits=num_bits)

    def to_tensor(self) -> torch.Tensor:
        qtensor = self.ctensor.to_tensor()
        scales = self.preprocess_cscales(
            cscales=self.cscales,
            shape=qtensor.shape)
        qscales = self.compute_qscales(
            scales=scales,
            num_bits=self.num_bits,
            unsigned=True,
            # This will cause graph breaks with `torch.compile`
            assert_values=False)
        # this should fail with integer tensors
        qscales = qscales.to(dtype=qtensor.dtype)
        return qtensor / qscales


@dataclass
class NFDoubleQuantizedTensorContainer(tensor_container_utils.TensorContainer):
    ctensor: PackedQuantizedTensorContainer
    cscales: UIntQuantizedTensorContainer
    offset: torch.Tensor
    num_bits: int
    qscheme: QuantScheme
    median_shape: Tuple[int, ...]

    @staticmethod
    def preprocess_cscales(
        cscales: UIntQuantizedTensorContainer,
        offset: torch.Tensor,
        final_shape: Tuple[int, ...],
        median_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        scales = cscales.to_tensor()
        scales = scales + offset
        scales = torch.broadcast_to(scales, median_shape)
        return scales.reshape(final_shape)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        cscales: UIntQuantizedTensorContainer,
        offset: torch.Tensor,
        num_bits: int,
        median_shape: torch.Size,
    ) -> Self:
        scales = cls.preprocess_cscales(
            cscales=cscales,
            offset=offset,
            final_shape=tensor.shape,
            median_shape=median_shape)
        if tensor.shape != scales.shape:
            raise ValueError
        qscheme = create_normal_float_scheme(
            num_bits=num_bits,
            device=tensor.device)

        tensor_scaled = tensor / scales
        # `-1` because this function assigns to the right bucket
        qtensor = torch.bucketize(
            tensor_scaled,
            qscheme.boundaries,
            right=False) - 1
        ctensor = PackedQuantizedTensorContainer.from_tensor(
            tensor=qtensor,
            num_bits=num_bits)
        return cls(
            ctensor=ctensor,
            cscales=cscales,
            offset=offset,
            num_bits=num_bits,
            qscheme=qscheme,
            median_shape=median_shape)

    # (TODO) check this entire function again
    def to_tensor(self) -> torch.Tensor:
        qtensor = self.ctensor.to_tensor()
        scales = self.preprocess_cscales(
            cscales=self.cscales,
            offset=self.offset,
            final_shape=qtensor.shape,
            median_shape=self.median_shape)
        tensor = self.qscheme.values[qtensor]
        tensor = tensor * scales
        return tensor

    # Hard-code the `from_dict` and `to_dict` methods for performance
    @classmethod
    def from_dict(cls, state_dict: Dict) -> Self:
        if cls is not NFDoubleQuantizedTensorContainer:
            raise ValueError
        return cls(
            ctensor=PackedQuantizedTensorContainer(
                ptensor=            state_dict["ctensor"]["ptensor"],
                padding_length=     state_dict["ctensor"]["padding_length"],
                shape=              state_dict["ctensor"]["shape"],
                num_bits=           state_dict["ctensor"]["num_bits"],
                dtype=              state_dict["ctensor"]["dtype"],
            ),
            cscales=UIntQuantizedTensorContainer(
                ctensor=PackedQuantizedTensorContainer(
                    ptensor=        state_dict["cscales"]["ctensor"]["ptensor"],
                    padding_length= state_dict["cscales"]["ctensor"]["padding_length"],
                    shape=          state_dict["cscales"]["ctensor"]["shape"],
                    num_bits=       state_dict["cscales"]["ctensor"]["num_bits"],
                    dtype=          state_dict["cscales"]["ctensor"]["dtype"],
                ),
                cscales=FloatTensorContainer(
                    qtensor=        state_dict["cscales"]["cscales"]["qtensor"],
                    qdtype=         state_dict["cscales"]["cscales"]["qdtype"],
                    dtype=          state_dict["cscales"]["cscales"]["dtype"],
                    num_bits=       state_dict["cscales"]["cscales"]["num_bits"],
                ),
                num_bits=           state_dict["cscales"]["num_bits"],
            ),
            offset=                 state_dict["offset"],
            num_bits=               state_dict["num_bits"],
            qscheme=                state_dict["qscheme"],
            median_shape=           state_dict["median_shape"],
        )

    def to_dict(self) -> Dict:
        return {
            "ctensor": {
                "ptensor":            self.ctensor.ptensor,
                "padding_length":     self.ctensor.padding_length,
                "shape":              self.ctensor.shape,
                "num_bits":           self.ctensor.num_bits,
                "dtype":              self.ctensor.dtype,
            },
            "cscales": {
                "ctensor": {
                    "ptensor":        self.cscales.ctensor.ptensor,
                    "padding_length": self.cscales.ctensor.padding_length,
                    "shape":          self.cscales.ctensor.shape,
                    "num_bits":       self.cscales.ctensor.num_bits,
                    "dtype":          self.cscales.ctensor.dtype,
                },
                "cscales": {
                    "qtensor":        self.cscales.cscales.qtensor,
                    "qdtype":         self.cscales.cscales.qdtype,
                    "dtype":          self.cscales.cscales.dtype,
                    "num_bits":       self.cscales.cscales.num_bits,
                },
                "num_bits":           self.cscales.num_bits,
            },
            "offset":                 self.offset,
            "num_bits":               self.num_bits,
            "qscheme":                self.qscheme,
            "median_shape":           self.median_shape,
        }


class NumBits(enum.Enum):
    # When compiled with dynamism, `torch.compile` will
    # treat `_num_bits` (integer) as a symbolic argument.
    # However, we want to treat this argument as a static
    # argument. So we use `NumBits` to wrap around it.
    NUM_BITS_2 = 2
    NUM_BITS_3 = 3
    NUM_BITS_4 = 4
    NUM_BITS_8 = 8


@dataclass
class NFDoubleQuantizedTensorContainer2(tensor_container_utils.TensorContainer):
    # This is an optimized version of the previous version. Note that
    # This version does not support quantizing Tensors. We rely on the
    # previous version to quantize Tensors, and convert them to this version.
    # The primary goal of this version is fast dequantization.
    block_sizes: Tuple[int, int, int]

    qscales_packed: packbits_utils.PackedBinaryTensorType
    qscales_offset: torch.Tensor
    qscales_qscales: torch.Tensor
    qscales_num_bits: NumBits

    qtensor_map: torch.Tensor
    qtensor_shape: Tuple
    qtensor_packed: packbits_utils.PackedBinaryTensorType
    qtensor_num_bits: NumBits

    @staticmethod
    def get_uint_max_bound(
        num_bits: int,
        unsigned: bool,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            (2.0 ** (num_bits - 1 + int(unsigned))) - 1.0,
            device=device)

    def to_tensor(self) -> torch.Tensor:

        # --------------- Second level dequantization ---------------
        # int32, [block_sizes[0], block_sizes[1], 1]
        qscales = packbits_utils.unpack_integer_tensors_2(
            packed_tensor=self.qscales_packed,
            num_bits=self.qscales_num_bits.value,
            shape=(self.block_sizes[0],
                   self.block_sizes[1],
                   1))  # expand last dimension

        # Computation must be in FP32 to prevent potential over flow.
        # float32, [block_sizes[0], block_sizes[1], 1]
        qscales = qscales.to(dtype=torch.float32)
        # float32, [block_sizes[0], 1, 1]
        qscales_scales = self.qscales_qscales.to(dtype=torch.float32)
        # float32, [block_sizes[0], block_sizes[1], 1]
        qscales_scales = torch.broadcast_to(
            qscales_scales,
            qscales.shape)

        max_bound = self.get_uint_max_bound(
            num_bits=self.qscales_num_bits.value,
            unsigned=True,
            device=qscales_scales.device)

        # qscales_offset + qscales * qscales_scales / max_bound
        # float32, [block_sizes[0], block_sizes[1], 1]
        scales = torch.addcmul(
            self.qscales_offset,
            qscales,
            qscales_scales,
            value=(1. / max_bound))

        # --------------- First level dequantization ---------------
        # int32, [*block_sizes]
        qtensor = packbits_utils.unpack_integer_tensors_2(
            packed_tensor=self.qtensor_packed,
            num_bits=self.qtensor_num_bits.value,
            shape=self.block_sizes)
        # float32, [*block_sizes]
        tensor = self.qtensor_map[qtensor]
        tensor = tensor * scales
        # float32, [*qtensor_shape]
        tensor = tensor.view(self.qtensor_shape)
        return tensor

    # Hard-code the `from_dict` and `to_dict` methods for performance
    @classmethod
    def from_dict(cls, state_dict: Dict) -> Self:
        if cls is not NFDoubleQuantizedTensorContainer2:
            raise ValueError
        return cls(
            block_sizes=            state_dict["block_sizes"],
            qscales_packed=         state_dict["qscales_packed"],
            qscales_offset=         state_dict["qscales_offset"],
            qscales_qscales=        state_dict["qscales_qscales"],
            qscales_num_bits=       state_dict["qscales_num_bits"],
            qtensor_map=            state_dict["qtensor_map"],
            qtensor_shape=          state_dict["qtensor_shape"],
            qtensor_packed=         state_dict["qtensor_packed"],
            qtensor_num_bits=       state_dict["qtensor_num_bits"],
        )

    def to_dict(self) -> Dict:
        return {
            "block_sizes":          self.block_sizes,
            "qscales_packed":       self.qscales_packed,
            "qscales_offset":       self.qscales_offset,
            "qscales_qscales":      self.qscales_qscales,
            "qscales_num_bits":     self.qscales_num_bits,
            "qtensor_map":          self.qtensor_map,
            "qtensor_shape":        self.qtensor_shape,
            "qtensor_packed":       self.qtensor_packed,
            "qtensor_num_bits":     self.qtensor_num_bits,
        }


def blockwise_absmax_nf_quantization(
    A: torch.Tensor,
    num_bits: int,
    num_bits_0: int,  # of the second-level quantization
    num_bits_1: str,  # of the second-level quantization states
    block_size_0: int,
    block_size_1: int,
) -> NFDoubleQuantizedTensorContainer:
    # (TODO) Double check this
    if A.dtype != torch.float32:
        raise ValueError(f"Expected float32, but got {A.dtype}")

    # Compute the second-level quantization
    A_reshaped = A.view(-1, block_size_1, block_size_0)
    scales_0 = dimwise_absmax(A_reshaped, dim=2)
    # Notice that we use the `.min` as the offset.
    # This guarantees that the the smallest number after
    # quantization will be at least `offset_0`, which is
    # positive because `scales_1` is non-negative.
    offset_0 = scales_0.min()
    scales_0_with_offset = scales_0 - offset_0
    scales_1 = dimwise_absmax(scales_0_with_offset, dim=1)

    # Compute the second-level quantization
    scales_1_container = FloatTensorContainer.from_tensor(
        tensor=scales_1,
        num_bits=num_bits_1)

    # Compute the first-level quantization
    scales_0_container = UIntQuantizedTensorContainer.from_tensor(
        tensor=scales_0_with_offset,
        cscales=scales_1_container,
        num_bits=num_bits_0)

    # (TODO) Double check this
    # Compute the final quantization
    return NFDoubleQuantizedTensorContainer.from_tensor(
        tensor=A,
        cscales=scales_0_container,
        offset=offset_0,
        num_bits=num_bits,
        median_shape=A_reshaped.shape)

def convert_v1_to_v2(container: NFDoubleQuantizedTensorContainer) -> NFDoubleQuantizedTensorContainer2:
    # if not isinstance(container, NFDoubleQuantizedTensorContainer):
    #     raise TypeError

    def _pack_to_int32(tensor: torch.Tensor, num_bits: int) -> packbits_utils.PackedBinaryTensorType:
        if tensor.dtype not in [torch.int64, torch.float32]:
            raise TypeError

        # ------- Safe casting -------
        # Explicit casting, and the following code will
        # raise an Error if casting leads to overflow
        bits_max = torch.tensor(
            2 ** 8 - 1,
            dtype=tensor.dtype,
            device=tensor.device)
        bits_min = torch.tensor(
            0,
            dtype=tensor.dtype,
            device=tensor.device)
        if tensor.max() > bits_max:
            raise ValueError
        if tensor.min() < bits_min:
            raise ValueError

        # packing requires tensors in `uint8` dtype
        tensor_uint8 = tensor.to(dtype=torch.uint8)
        if not (tensor_uint8.to(dtype=tensor.dtype) == tensor).all():
            raise ValueError
        return packbits_utils.pack_integer_tensors_2(
            tensor=tensor_uint8,
            num_bits=num_bits)

    # simply materialize and re-pack these tensors
    qtensor_packed = _pack_to_int32(
        tensor=container.ctensor.to_tensor(),
        num_bits=container.ctensor.num_bits)
    qscales_packed = _pack_to_int32(
        tensor=container.cscales.ctensor.to_tensor(),
        num_bits=container.cscales.ctensor.num_bits)

    # [-1, block_size_1, block_size_0]
    block_sizes = container.median_shape
    if len(block_sizes) != 3:
        raise ValueError
    # breakpoint()
    if container.ctensor.dtype != torch.int64:
        raise ValueError
    if math.prod(container.ctensor.shape) != math.prod(block_sizes):
        raise ValueError
    if container.cscales.ctensor.dtype != torch.float32:
        raise ValueError
    if container.cscales.ctensor.shape != (block_sizes[0], block_sizes[1], 1):
        raise ValueError
    if container.cscales.cscales.dtype != torch.float32:
        raise ValueError
    if container.cscales.cscales.qtensor.shape != (block_sizes[0], 1, 1):
        raise ValueError

    return NFDoubleQuantizedTensorContainer2(
        block_sizes=block_sizes,
        qscales_packed=qscales_packed,
        qscales_offset=container.offset,
        qscales_qscales=container.cscales.cscales.qtensor,
        qscales_num_bits=NumBits(container.cscales.ctensor.num_bits),
        qtensor_map=container.qscheme.values,
        qtensor_shape=container.ctensor.shape,
        qtensor_packed=qtensor_packed,
        qtensor_num_bits=NumBits(container.ctensor.num_bits))


# @torch.compile
def quantize(
    A: torch.Tensor,
    method: str,
    qconfig: QuantConfig,
    legacy: bool = True,  # TODO: change that if it is too slow... but we need to create NFDoubleQuantizedTensorContainerBucket2 ...
) -> tensor_container_utils.QuantizedTensor:

    misc_utils.swarn(
        f"Quantization scheme: "
        f"method={method}, "
        f"qconfig={qconfig}",
        fg="red")

    if method == "blockwise-nf":
        ctensor = blockwise_absmax_nf_quantization(
            A,
            num_bits=qconfig.num_bits,
            num_bits_0=qconfig.num_bits_0,
            num_bits_1=qconfig.num_bits_1,
            block_size_0=qconfig.block_size_0,
            block_size_1=qconfig.block_size_1)
    elif method == "blockwise-permute":
        ctensor = blockwise_permute_quantization(
            A,
            num_bits=qconfig.num_bits,
            num_bits_0=qconfig.num_bits_0,
            num_bits_1=qconfig.num_bits_1,
            block_size_0=qconfig.block_size_0,
            block_size_1=qconfig.block_size_1,
            bucket=2)      #Or =4

    if legacy is False:
        ctensor = convert_v1_to_v2(ctensor)

    return tensor_container_utils.QuantizedTensor(
        container=ctensor,
        tensor_like=A,  # this tensor behaves like `A`
        transpose=False,
        compute_dtype=None)

    raise ValueError


def patch_qtensor_for_fast_dequantization_(
    qtensor: tensor_container_utils.QuantizedTensor,
) -> None:
    if not isinstance(qtensor, tensor_container_utils.QuantizedTensor):
        raise TypeError
    if not isinstance(qtensor._container, NFDoubleQuantizedTensorContainer):
        raise TypeError
    if not all([
        qtensor._transpose is False,
        qtensor._compute_dtype is None]):
        raise ValueError

    qtensor._container = convert_v1_to_v2(
        container=qtensor._container)


@dataclass
class NFDoubleQuantizedTensorContainerBucket(tensor_container_utils.TensorContainer):
    ctensor: PackedQuantizedTensorContainer
    cscales: UIntQuantizedTensorContainer
    offset: torch.Tensor
    num_bits: int
    median_shape: Tuple[int, ...]
    cb: torch.Tensor
    bucket: int
    permut: list

    @staticmethod
    def preprocess_cscales(
            cscales: UIntQuantizedTensorContainer,
            offset: torch.Tensor,
            final_shape: Tuple[int, ...],
            median_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        scales = cscales.to_tensor()
        scales = scales + offset
        scales = torch.broadcast_to(scales, median_shape)
        return scales.reshape(final_shape)

    @classmethod
    def from_tensor(
            cls,
            tensor: torch.Tensor,
            cscales: UIntQuantizedTensorContainer,
            offset: torch.Tensor,
            num_bits: int,
            median_shape: torch.Size,
            bucket: int,
            permut: list,
            cb: torch.Tensor,
    ) -> Self:
        scales = cls.preprocess_cscales(
            cscales=cscales,
            offset=offset,
            final_shape=tensor.shape,
            median_shape=median_shape)
        if tensor.shape != scales.shape:
            raise ValueError

        ###
        ###
        if bucket > 2 and (tensor.shape[0] > 5120 or tensor.shape[1] > 5120):
            cut = tensor.shape[0] // bucket // 8 # TODO if cuda out of memory divide by 4 or even 8
        else:
            cut = tensor.shape[0] // bucket // 2  # needed for 13b
        ###
        ###
        codebook = cb.unsqueeze(0).unsqueeze(3).expand(cut, bucket, 2 ** (num_bits * bucket), tensor.shape[1]).to(
            tensor.device)

        tensor_scaled = tensor / scales
        tensor_scaled = tensor_scaled.view(-1, bucket, tensor.shape[1])
        L = list()
        for k in range(tensor_scaled.shape[0] // cut): # (bucket * cut)):
            # B = tensor_scaled[k * bucket * cut:(k + 1) * bucket * cut, :, :].to(tensor.device)
            B = tensor_scaled[k * cut:(k + 1) * cut, :, :].to(tensor.device)
            # diff = torch.unsqueeze(B, 2) - codebook
            # diff_norm = torch.linalg.norm(torch.unsqueeze(B, 2) - codebook, ord=2, dim=1, keepdim=True)
            # diff_min = torch.argmin(torch.linalg.norm(torch.unsqueeze(B, 2) - codebook, ord=2, dim=1, keepdim=True), dim=2, keepdim=True)
            index_ppv_s = torch.argmin(torch.linalg.norm(torch.unsqueeze(B, 2) - codebook, ord=2, dim=1, keepdim=True), dim=2, keepdim=True).expand(cut, 1, 1, tensor.shape[1])
            del B
            L.append(index_ppv_s.squeeze())
        qtensor = torch.cat(L, 0)

        ctensor = PackedQuantizedTensorContainer.from_tensor(
            tensor=qtensor,
            num_bits=num_bits * bucket)
        del qtensor
        del codebook
        del tensor
        del tensor_scaled
        del scales
        del index_ppv_s
        del L
        torch.cuda.empty_cache()
        return cls(
            ctensor=ctensor,
            cscales=cscales,
            offset=offset,
            num_bits=num_bits,
            median_shape=median_shape,
            cb=cb, # codebook,
            bucket=bucket,
            permut=permut)

    # layer_q.self_attn.q_proj.base_layer.weight._container.cscales.cscales.qtensor.requires_grad = True
    # layer_q.self_attn.q_proj.base_layer.weight._container.cscales.cscales.qtensor.requires_grad = True
    # layer_q.self_attn.q_proj.base_layer.weight._container.cb.requires_grad = True

    # (TODO) check this entire function again
    def to_tensor(self) -> torch.Tensor:
        qtensor = self.ctensor.to_tensor().to(self.ctensor.ptensor.dtype)
        scales = self.preprocess_cscales(
            cscales=self.cscales,
            offset=self.offset,
            final_shape=(qtensor.shape[0] * self.bucket, qtensor.shape[1]),
            median_shape=self.median_shape)
        ###
        ###
        if self.bucket > 2 and (qtensor.shape[0] > 3000 or qtensor.shape[1] > 5120):
            cut = qtensor.shape[0] // 4
        else:
            cut = qtensor.shape[0]
        ###
        ###
        L = list()
        codebook = self.cb.unsqueeze(0).unsqueeze(3).expand(cut, self.bucket, 2 ** (self.num_bits * self.bucket), qtensor.shape[1]).to(qtensor.device)            # 1024 x 4 x 256 x 4096


        # for k in range(scales.shape[0] // (self.bucket * cut)):
        for k in range(qtensor.shape[0] // cut):
            # index_ppv_s = qtensor[k * self.bucket * cut:(k + 1) * self.bucket * cut, :].unsqueeze(1).unsqueeze(
            index_ppv_s = qtensor[k * cut:(k + 1) * cut, :].unsqueeze(1).unsqueeze(2).expand(cut, self.bucket, 1, qtensor.shape[1])
            dq = torch.gather(codebook, dim=2, index=index_ppv_s.to(torch.int64))
            L.append(dq.squeeze())
            # torch.cuda.empty_cache()

        tensor = torch.cat(L, 0)
        tensor = tensor.view(scales.shape[0], scales.shape[1]) * scales
        L = list()
        for k in range(tensor.shape[0] // 64):
            # print(self.permut[k])
            L.append(tensor[64 * k:64 * (k + 1), self.permut[k].long()]) #[self.permut[k].index(i) for i in range(tensor.shape[1])]])
            # L.append(tensor[64 * k:64 * (k + 1), :])
        #del tensor
        #del scales
        #del codebook
        #del qtensor
        #torch.cuda.empty_cache()
        return torch.cat(L, 0)

    ###############
    ##############
    ###############
    # Hard-code the `from_dict` and `to_dict` methods for performance
    @classmethod
    def from_dict(cls, state_dict: Dict) -> Self:
        if cls is not NFDoubleQuantizedTensorContainerBucket:
            raise ValueError
        return cls(
            ctensor=PackedQuantizedTensorContainer(
                ptensor=state_dict["ctensor"]["ptensor"],
                padding_length=state_dict["ctensor"]["padding_length"],
                shape=state_dict["ctensor"]["shape"],
                num_bits=state_dict["ctensor"]["num_bits"],
                dtype=state_dict["ctensor"]["dtype"],
            ),
            cscales=UIntQuantizedTensorContainer(
                ctensor=PackedQuantizedTensorContainer(
                    ptensor=state_dict["cscales"]["ctensor"]["ptensor"],
                    padding_length=state_dict["cscales"]["ctensor"]["padding_length"],
                    shape=state_dict["cscales"]["ctensor"]["shape"],
                    num_bits=state_dict["cscales"]["ctensor"]["num_bits"],
                    dtype=state_dict["cscales"]["ctensor"]["dtype"],
                ),
                cscales=FloatTensorContainer(
                    qtensor=state_dict["cscales"]["cscales"]["qtensor"],
                    qdtype=state_dict["cscales"]["cscales"]["qdtype"],
                    dtype=state_dict["cscales"]["cscales"]["dtype"],
                    num_bits=state_dict["cscales"]["cscales"]["num_bits"],
                ),
                num_bits=state_dict["cscales"]["num_bits"],
            ),
            offset=state_dict["offset"],
            num_bits=state_dict["num_bits"],
            median_shape=state_dict["median_shape"],
            cb=state_dict["cb"],
            bucket=state_dict["bucket"],
            permut=state_dict["permut"],
        )

    def to_dict(self) -> Dict:
        return {
            "ctensor": {
                "ptensor": self.ctensor.ptensor,
                "padding_length": self.ctensor.padding_length,
                "shape": self.ctensor.shape,
                "num_bits": self.ctensor.num_bits,
                "dtype": self.ctensor.dtype,
            },
            "cscales": {
                "ctensor": {
                    "ptensor": self.cscales.ctensor.ptensor,
                    "padding_length": self.cscales.ctensor.padding_length,
                    "shape": self.cscales.ctensor.shape,
                    "num_bits": self.cscales.ctensor.num_bits,
                    "dtype": self.cscales.ctensor.dtype,
                },
                "cscales": {
                    "qtensor": self.cscales.cscales.qtensor,
                    "qdtype": self.cscales.cscales.qdtype,
                    "dtype": self.cscales.cscales.dtype,
                    "num_bits": self.cscales.cscales.num_bits,
                },
                "num_bits": self.cscales.num_bits,
            },
            "offset": self.offset,
            "num_bits": self.num_bits,
            "median_shape": self.median_shape,
            "cb": self.cb,
            "bucket": self.bucket,
            "permut": self.permut,
        }

def train_codebook_opt_mat_cpu(X, dimension, Ks, train_size=1000000, iter=20):
    X = np.array(X, dtype=np.float32)
    codewords, _ = kmeans2(X, Ks, iter=iter, minit='points')
    return codewords


def blockwise_permute_quantization(
        A: torch.Tensor,
        num_bits: int,
        num_bits_0: int,  # of the second-level quantization
        num_bits_1: str,  # of the second-level quantization states
        block_size_0: int,
        block_size_1: int,
        bucket: int,
) -> NFDoubleQuantizedTensorContainerBucket:
    # (TODO) Double check this
    if A.dtype != torch.float32:
        raise ValueError(f"Expected float32, but got {A.dtype}")

    L = list()
    scales = list()
    # lrdp = list()
    permutations_list = list()
    for k in tqdm(range(A.shape[0] // 64)):
        I, ind = bloc_permute(A, 64, 16, k)
        # I = A[64*k: 64*(k+1), :]
        # ind = torch.arange(A.shape[1]).to(torch.int64)
        L.append(I)
        # lrdp.append(A[64 * k:64 * (k + 1), ind])
        permutations_list.append(ind.to(torch.int16))
    cat = torch.cat(L, 0)
    # lrdiff_permut = torch.cat(lrdp, 0)

    _, scales_dq = blockwise_absmax(cat, 8, "fp32", block_size_0, block_size_1)
    A_scaled = cat / scales_dq
    A_scaled = A_scaled.view(-1, bucket, A.shape[1])
    B = torch.permute(A_scaled, (0, 2, 1))

    begining = time.time()
    optimal_codebook = train_codebook_opt_mat_cpu(B.reshape(-1, bucket).detach().cpu().numpy(), bucket,
                                                  2 ** (bucket * num_bits)).T
    end = time.time() -begining
    print(f'total time for codebook: {end:.2f}s')

    A = cat

    # Compute the second-level quantization
    A_reshaped = A.view(-1, block_size_1, block_size_0)
    scales_0 = dimwise_absmax(A_reshaped, dim=2)
    # Notice that we use the `.min` as the offset.
    # This guarantees that the the smallest number after
    # quantization will be at least `offset_0`, which is
    # positive because `scales_1` is non-negative.
    offset_0 = scales_0.min()
    scales_0_with_offset = scales_0 - offset_0
    scales_1 = dimwise_absmax(scales_0_with_offset, dim=1)

    # Compute the second-level quantization
    begining = time.time()
    scales_1_container = FloatTensorContainer.from_tensor(
        tensor=scales_1,
        num_bits=num_bits_1)
    end = time.time() -begining
    print(f'total time for FloatTensorContainer: {end:.2f}s')

    # Compute the first-level quantization
    begining = time.time()
    scales_0_container = UIntQuantizedTensorContainer.from_tensor(
        tensor=scales_0_with_offset,
        cscales=scales_1_container,
        num_bits=num_bits_0)
    end = time.time() -begining
    print(f'total time for UIntQuantizedTensorContainer: {end:.2f}s')

    # (TODO) Double check this
    # Compute the final quantization
    begining = time.time()
    #breakpoint()
    print('optimal codebook', torch.tensor(optimal_codebook).shape)
    # breakpoint()
    outs = NFDoubleQuantizedTensorContainerBucket.from_tensor(
        tensor=A,
        cscales=scales_0_container,
        offset=offset_0,
        num_bits=num_bits,
        median_shape=A_reshaped.shape,
        bucket=bucket,
        permut=permutations_list,
        cb=torch.tensor(optimal_codebook))
    end = time.time() -begining
    print(f'total time for NFDoubleQuantizedTensorContainerBucket: {end:.2f}s')
    return outs


def bloc_permute(T, cut, overlap, blocindex):
    Im = T[blocindex * cut:(blocindex + 1) * cut, :]

    ### This matrice is not getting to be column-permuted!
    Imbase = T[blocindex * cut:(blocindex + 1) * cut, :]

    # list_indexes = [0]
    inverted_indices = torch.zeros(T.shape[1], dtype=torch.long)
    for i in range(T.shape[1] - 3):
        index_ppv = torch.argmin(
            torch.linalg.norm(Im[:, i - overlap:i].mean(dim=1).unsqueeze(dim=1) - Im[:, i + 1:], ord=2, dim=0,
                              keepdim=True))
        kept_index = torch.topk(
            -torch.linalg.norm(Imbase - Im[:, i + index_ppv + 1].unsqueeze(1).expand_as(Imbase), ord=2, dim=0,
                               keepdim=True), 1).indices.squeeze()
        # kept_index = i + 1

        # list_indexes.append(int(kept_index))
        inverted_indices[kept_index] = i + 1

        if index_ppv != 0:
            if index_ppv == Im[:, i + 1:].shape[1] - 1:
                Im = Im[:, torch.cat(
                    (torch.range(0, i, device=index_ppv.device), (i + index_ppv + 1).unsqueeze(dim=0),
                     torch.range(i + 1, i + index_ppv, device=index_ppv.device))).to(
                    dtype=torch.long)]
            else:

                Im = Im[:, torch.cat((torch.range(0, i, device=index_ppv.device), (i + index_ppv + 1).unsqueeze(dim=0),
                                      torch.range(i + 1, i + index_ppv, device=index_ppv.device),
                                      torch.range(i + index_ppv + 2, T.shape[1] - 1, device=index_ppv.device))).to(dtype=torch.long)]
    #alone = list()
    #for i in tqdm(range(1, Imbase.shape[1]), desc='line 903'):
    #    if i not in [int(e) for e in list_indexes]:
    #        alone.append(i)
    for i in [T.shape[1]-2, T.shape[1]-1]:
        kept_index = torch.topk(
            -torch.linalg.norm(Imbase - Im[:, i].unsqueeze(1).expand_as(Imbase), ord=2, dim=0,
                               keepdim=True), 1).indices.squeeze()
        # list_indexes.append(int(kept_index))
        inverted_indices[kept_index] = i
    return Im, inverted_indices.to(torch.int16)  # list_indexes