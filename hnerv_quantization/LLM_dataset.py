import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM
from typing import Tuple

class LLMDataset(Dataset):
    def __init__(self, model_path, layer_depth=0, layer_type='q', p_size=512, rank=64, do_permute=False):
        if '_' in str(layer_depth):
            self.layer_depth = [int(layer_d) for layer_d in str(layer_depth).split('_')]
        elif type(layer_depth) == list:
            self.layer_depth = layer_depth
        else:
            self.layer_depth = [layer_depth]
        self.layer_type = layer_type
        self.rank = rank
        self.p_size = p_size
        self.final_size = p_size**2
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.img = []  # some array of images
        self.all_depths = []
        self.coord = []
        self.do_permute = do_permute
        self.A, self.L1, self.L2, self.scales_q = {}, {}, {}, {}
        self.idx = []
        block = -1
        for i, layer in enumerate(model.model.layers):
            modules = get_modules(layer, "llama")
            module_names = get_module_names("llama")

            for lin, name in zip(modules, module_names):
                if name == layer_type:
                    block += 1
                    if block in self.layer_depth:
                        self.A[block] = lin.weight.data
                        self.H, self.W = self.A[block].shape
                        self.L1[block], self.L2[block] = svd_decomposition(self.A[block], randomized=True, num_ranks=rank, num_oversampling=10)
                        A = self.A[block] - self.L1[block]@self.L2[block]
                        if do_permute:
                            A_scaled, Index, Gains = do_permute_max(A, 64, 16, False)
                            # to_save = (to_save + 1) / 2
                        else:
                            _, self.scales_q[block] = blockwise_absmax(A,"fp32", 64, 256)
                            A_scaled = A / self.scales_q[block]
                        A_scaled = (A_scaled + 1) * (1 / 2)
                        idx = 0
                        for j in range(int(self.H//p_size)):
                            for k in range(int(self.W//p_size)):
                                patch = A_scaled[p_size * j:p_size * (j + 1), p_size * k:p_size * (k + 1)]
                                patch = torch.unsqueeze(patch, 0)
                                self.img.append(patch)
                                self.coord.append(torch.tensor([j, k]))
                                self.idx.append(idx)
                                idx += 1
                                self.all_depths.append(block)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        idx = self.idx[item]
        layer_depth = self.all_depths[item]
        sample = {'img': self.img[item], 'idx': idx, 'norm_idx': idx / self.__len__(), 'layer_depth': layer_depth}
        return sample

    def reconstruct(self, im_lst, coords, layer_depth):
        A_hat = torch.zeros_like(self.A[layer_depth])
        for im, (j, k) in zip(im_lst, coords):
            # im must be p_size x p_size
            A_hat[self.p_size * j:self.p_size * (j + 1), self.p_size * k:self.p_size * (k + 1)] = im
        A_hat = A_hat * 2 - 1
        A_hat = A_hat * self.scales_q[layer_depth]
        return A_hat + self.L1[layer_depth] @ self.L2[layer_depth]


def bloc_permute_max(T, cuting, overlap, rankfirst, blocindex):
    cut=cuting
    Im = T[blocindex*cut:(blocindex+1)*cut,:]
    gains = torch.max(Im.abs(), dim=0, keepdim=True).values
    Im = Im /gains

    ### This matrice is not getting to be column-permuted!
    Imbase = T[blocindex*cut:(blocindex+1)*cut,:] / gains

    if rankfirst:
       values, indices = torch.topk(Im[:,0], Im.shape[0])
       Im = Im[indices, :]
    list_indexes = list()
    for i in range(4093):
        index_ppv = torch.argmin(torch.linalg.norm(Im[:,i-overlap:i].mean(dim=1).unsqueeze(dim=1)-Im[:,i+1:], ord=2, dim=0, keepdim=True))
        list_indexes.append(torch.topk(-torch.linalg.norm(Imbase - Im[:,i+index_ppv+1].unsqueeze(1).expand_as(Imbase), ord=2, dim=0, keepdim=True),1).indices.squeeze())
        #list_indexes.append(i+index_ppv+1)

        if index_ppv != 0:
            if index_ppv == Im[:,i+1:].shape[1]-1:
                Im = Im[:, torch.cat((torch.range(0, i), (i+index_ppv+1).unsqueeze(dim=0), torch.range(i+1, i+index_ppv))).to(dtype=torch.long)]
            else:
                Im = Im[:, torch.cat((torch.range(0, i), (i+index_ppv+1).unsqueeze(dim=0), torch.range(i+1, i+index_ppv), torch.range(i+index_ppv+2, 4095))).to(dtype=torch.long)]
    return Im, list_indexes, gains.expand_as(Im)

def do_permute_max(T, cuting, overlap, rankfirst):
    L = list()
    idlist = list()
    G = list()
    for k in range(64):
        print(k)
        I, ind, g = bloc_permute_max(T, cuting, overlap, rankfirst, k)
        L.append(I)
        idlist.append(ind)
        G.append(g)
    T = torch.cat(L,0)
    return T, idlist, torch.cat(G,0)



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


def dimwise_absmax(A: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.max(
        torch.abs(A),
        dim=dim,
        keepdim=True).values
def blockwise_absmax(
    A: torch.Tensor,
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
    # This guarantees that the the smallest number after
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

    scales_2_ = scales_2
    scales_1_ = scales_2_ + offset_1

    scales_q = torch.broadcast_to(scales_1, scales_0.shape)
    scales_dq = torch.broadcast_to(scales_1_, scales_0.shape)
    scales_q = scales_q.reshape(A.shape)
    scales_dq = scales_dq.reshape(A.shape)
    return scales_q, scales_dq


def get_module_names(model_type):
    if model_type == "opt":
        return ["q", "k", "v", "o", "up", "down"]
    else:
        assert model_type == "llama"
        return ["q", "k", "v", "o", "gate", "up", "down"]


def get_modules(layer, model_type):
    if model_type == "opt":
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
            layer.fc1,
            layer.fc2,
        ]
    else:
        # llama or vicuna
        assert model_type == "llama"
        modules = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ]

    return modules
