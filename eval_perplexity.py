import copy
import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel, AutoModelForCausalLM

from src.datautils import get_loaders, set_seed

from src.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
    save_not_quantized_weights,
)
from src.utils import using_tf32
from models import lora_utils

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model, args):
    """main entry point to functions for model quantization"""
    tick = time.time()
    results = quantize_aq(model, args)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(
    model: PreTrainedModel, data_iterable: Iterable, args: Namespace, nsamples: Optional[int] = None
) -> Sequence[torch.Tensor]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples or len(data_iterable)
    device = args.devices[0] if not args.offload_activations else torch.device("cpu")
    assert nsamples is not None

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    try:
        layers[0] = layers[0].to(device)
    except NotImplementedError:
        ()

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (nsamples - 1) // len(args.devices) + 1
    print('model.seqlen', model.seqlen)
    inps = [
        torch.zeros(
            (min(nsamples_per_device, nsamples - i * nsamples_per_device), model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device=args.devices[i] if not args.offload_activations else "cpu",
            pin_memory=args.offload_activations,
        )
        for i in range(len(args.devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1

            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data_iterable:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # print('batch_inps', batch_inps.shape)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}

    return inps, forward_args


@torch.no_grad()
def quantize_aq(
    model: PreTrainedModel, args: Namespace
):
    assert not torch.backends.cuda.matmul.allow_tf32
    '''
    print("\nStarting AQ quantization ...")
    inps, forward_args = get_inps(model, dataloader, args)  # here the model is not quantized
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]

    if val_dataloader:
        run_validation = True
        val_inps, _ = get_inps(model, val_dataloader, args)
        val_outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps]
    else:
        run_validation = False
        val_inps, val_outs = None, None
    '''
    model_q = copy.deepcopy(model)

    model_q = lora_utils.load_peft_model(model_q, '/'.join(args.checkpoint_dir.split('/')[:-1]))
    print(model_q)
    # use_cache = model_q.config.use_cache
    model_q.config.use_cache = False

    layers_quantized = get_layers(model_q)

    # use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_layers(model)
    print(model)
    # =============== block-wise finetuning like in AQLM ================== #
    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")


        # quantized layer will return there
        mean_change = model_q.base_model.model.model.layers[layer_index].self_attn.q_proj.lora_A.default.weight.data.mean()
        layer_q = layers_quantized[layer_index].to(args.devices[0])
        layer_save_path = os.path.join(args.checkpoint_dir, f"{layer_index}.pth")
        layer_q.load_state_dict(torch.load(layer_save_path))

        mean_change -= model_q.base_model.model.model.layers[layer_index].self_attn.q_proj.lora_A.default.weight.data.mean()
        print(mean_change)

    return model_q


@torch.no_grad()
def perplexity_eval(model, testenc, args):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")
    if 'c4' in args.dataset_name:
        seqlen = 2048  # 1024
    elif 'wiki' in args.dataset_name:
        seqlen = 2048
    else:
        seqlen = 2048
    nsamples = testenc.numel() // seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.seqlen = seqlen
    print('testenc', testenc.shape)
    inps, forward_args = get_inps(model, testenc, args, nsamples=nsamples)
    print('inps', inps[0].shape)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    device = args.devices[0]
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        else:
            update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
        try:
            layers[i] = layer.cpu()
        except:
            ()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])

    nlls = []
    for i in range(nsamples):
        inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache

@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        # breakpoint()
        #print(inps_tensor[j].to(device).unsqueeze(0).shape)
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            batch_size = outs_batch.shape[0]
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device)).float().square().view(batch_size, -1).mean(dim=-1)
            )
            outs_batch_loss /= outs_batch.float().square().view(batch_size, -1).mean(dim=-1).clamp(min=1e-6)
            outs_batch_loss = outs_batch_loss.mean()
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--model_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='c4',
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default='/mnt/data/lisa/LLM_results/VQ_dora_2b_bucket2_rank64', # /finetune_bloc # /finetune_global
        help="Initialization of the quantized model weights",
    )
    parser.add_argument(
        "--finetune_type",
        type=str,
        default='bloc',
        help="Finetuning type: layer or bloc wise.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024, # 1024
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,  # 4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--use_checkpointing",
        action="store_true",
        help="Whether to use checkpoining in finetuning",
    )
    parser.add_argument(
        "--mix_compression",
        action="store_true",
        help="Compress .self_attn in 4 bits, .block_sparse_moe.experts to 2.3 for mixtral.",
    )
    parser.add_argument("--quantizor_type", type=str, default='dynamo', help="classic, dora or dynamo")
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None, help="List of devices")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=3,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--scale_nbits",
        type=int,
        default=0,
        help="Number of bits dedicated to the learnable group-wise scale. 0 means do not use group-wise scales "
        "(still has row-wise scales), 1-15 means using per-group scales quantized to this many bits, "
        "16+ means use per-group scales but do not quantize them",
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits",
    )
    parser.add_argument(
        "--codebook_value_num_groups",
        type=int,
        default=1,
        help="Split codebook vectors into this many groups for quantizations. Only used when quantized codebooks.",
    )

    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Whether to use faiss.Kmeans when initializing codebooks and codes",
    )
    parser.add_argument(
        "--init_max_points_per_centroid",
        type=int,
        default=None,
        help="During K-means initialzation, sample (this_many * 2 ^ nbits_per_codebook) points for training K-means",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Maximum number of beam search rounds before the optimization is forcibly stopped.",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=0.01,
        help="Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=50,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=32,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=1,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="Num validation sequences",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Skip model quantization and immediately evaluate the loaded model",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "flash_attention_2", "sdpa"],
        help="Attention implementation.",
    )

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    set_seed(args.seed)

    args.offload_activations = True
    if args.devices is None:
        args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)

    # validate val size
    if args.nsamples is not None:
        assert args.val_size < args.nsamples, "Number of validation set must be smaller than train + val"

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    print("\n============ Load model... ============")
    # model = get_model(
    #     args.model_path, args.load, args.dtype, 4096, attn_implementation=args.attn_implementation
    # ).train(False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).train(False)
    model.seqlen = args.model_seqlen

    if 'finetune_bloc' in args.checkpoint_dir.split('/')[-1]:
        print("\n============ loading model which has been finetuned by blocs... ============")
        model = quantize_model(model, args)
        print('###############################')
        print(model)
        print('###############################')

    else:
        print("\n============ loading model which hasnt been finetuned or has been finetuned globally... ============")
        model = lora_utils.load_peft_model(model, args.checkpoint_dir)
        print(model)
    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]
    print('###############################')
    print(model)
    print('###############################')
    # breakpoint()
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=2048,# to be able to benchmark with litterature
            eval_mode=True,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
