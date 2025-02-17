#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import click
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, Any

import datasets
import evaluate
import torch
from datasets import (
    load_dataset,
    concatenate_datasets)
from src.modelutils import get_layers

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from models import misc_utils
from models import lora_utils
from models import distributed_utils
from experiments import mmlu_utils
from experiments import callback_utils


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    checkpoint_dir: Optional[str] = field(
        default='/mnt/data/lisa/LLM_results/VQ_dora3_8B_2b_bucket4_rank64', # VQ_dora13_2b_bucket4_rank64',
        metadata={
            "help": (
                "Loading already quantized model"
            )
        },
    )
    block_wise: bool = field(
        default=False,
        metadata={"help": "Wether to load the fintuned blocks or not."},
    )
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",  # "meta-llama/Llama-2-13b-hf",
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    method: str = field(
        default="blockwise-permute",
        metadata={"help": "Method used for quantization.",
                  "choices": ["blockwise-nf", "blockwise-permute"],},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-clbi login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    hf_quantization_method: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default='c4', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        # default='/mnt/data/lisa/LLM_results/VQ_dora13_3b_bucket2_rank64/finetune_bloc/finetune_global/checkpoint-295',
        default=None, # '/mnt/data/lisa/LLM_results/VQ_dora_2b_bucket2_rank64/finetune_bloc/finetune_global/checkpoint-501',
        metadata={
            "help": (
                "For starting the training from a specific checkpoint."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=2048, # 1024 ??
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main(return_trainer: bool = False):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments)) #, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args, data_args = parser.parse_args_into_dataclasses()

    output_dir = os.path.join(model_args.checkpoint_dir, 'finetune_bloc/finetune_global')
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(output_dir=output_dir, bf16=True, num_train_epochs=0.5,
                                      per_device_train_batch_size=1,
                                      gradient_accumulation_steps=32, evaluation_strategy='no',
                                      resume_from_checkpoint=data_args.resume_from_checkpoint,
                                      save_steps=0.3, # 0.1,
                                      save_total_limit=5,
                                      learning_rate=2e-5, weight_decay=0.,
                                      warmup_ratio=0.03, lr_scheduler_type='cosine',
                                      logging_steps=1, do_train=True,
                                      do_eval=False, tf32=True,
                                      overwrite_output_dir=True)
    # previous_checkpoint

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        elif training_args.resume_from_checkpoint is not None:
            last_checkpoint=training_args.resume_from_checkpoint

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name == "c4":
        misc_utils.swarn(
            f"Using C4 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "allenai/c4",
            'default',
            # "allenai--c4",
            data_files={
                "train": "en/c4-train.00000-of-01024.json.gz",  # MDDIF to have a different partition than the block wise finetuning
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        _wikitext_dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset = (
            wikitext_dataset  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset["text"])
            .add_column(
                name="url",
                column=wikitext_dataset["text"])
        )
        raw_datasets["wikitext"] = wikitext_dataset
    elif data_args.dataset_name == "c4-wiki-large":
        misc_utils.swarn(
            f"Using C4+WikiText2 dataset (`dataset_name` "
            f"= {data_args.dataset_name})",
            bg="yellow")
        raw_datasets = load_dataset(
            "allenai/c4",
            #"allenai--c4",
            'default',
            data_files={
                "train": [
                    "en/c4-train.00000-of-01024.json.gz",
                    "en/c4-train.00001-of-01024.json.gz"],
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
        )
        _wikitext_dataset_train = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train")
        _wikitext_dataset_eval = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test")
        # Hacks to be consistent with other works' preprocessing.
        wikitext_dataset_train = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_train["text"])
                ],
            },
        )
        wikitext_dataset_eval = datasets.Dataset.from_dict(
            {
                "text": [
                    # https://github.com/IST-DASLab/gptq/blob/main/datautils.py#L10
                    "\n\n".join(_wikitext_dataset_eval["text"])
                ],
            },
        )
        # Hacks to get around the `remove_columns` to be used later.
        wikitext_dataset_train = (
            wikitext_dataset_train  # type: ignore
            .add_column(
                name="timestamp",
                column=[None for _ in range(len(wikitext_dataset_train["text"]))])
            .add_column(
                name="url",
                column=wikitext_dataset_train["text"])
        )
        wikitext_dataset_eval = (
            wikitext_dataset_eval  # type: ignore
            .add_column(
                name="timestamp",
                column=wikitext_dataset_eval["text"])
            .add_column(
                name="url",
                column=wikitext_dataset_eval["text"])
        )
        raw_datasets["train"] = concatenate_datasets([
            raw_datasets["train"],
            wikitext_dataset_train])
        raw_datasets["wikitext"] = wikitext_dataset_eval
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                streaming=data_args.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        hf_quantization_kwargs = lora_utils.get_hf_quantization_config(
            method=model_args.hf_quantization_method,
            sequence_length=data_args.block_size)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            **hf_quantization_kwargs)
        # https://github.com/huggingface/transformers/pull/24906
        if model.config.pretraining_tp != 1:
            raise NotImplementedError
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those glpqroups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # ========= loading the model that has been fintuned by bloc =========== #
    model = lora_utils.load_peft_model(model, model_args.checkpoint_dir)
    print(model)
    if model_args.block_wise:
        model.config.use_cache = False

        layers_quantized = get_layers(model)
        for layer_index in range(len(layers_quantized)):
            print(f"\n---------------- Layer {layer_index} of {len(layers_quantized)} ----------------")


            # quantized layer will return there
            mean_change = model.base_model.model.model.layers[layer_index].self_attn.q_proj.lora_A.default.weight.data.mean()
            layer_q = layers_quantized[layer_index]
            layer_save_path = os.path.join(model_args.checkpoint_dir,
                                           f"finetune_bloc/{layer_index}.pth")
            layer_q.load_state_dict(torch.load(layer_save_path))
            mean_change -= model.base_model.model.model.layers[
                layer_index].self_attn.q_proj.lora_A.default.weight.data.mean()
            print(mean_change)


    # This has no effects if we are not using DDP. But if we are, this
    # will patch the model to be compatible with DDP.
    distributed_utils.maybe_prepare_model_for_ddp(
        args=training_args,
        model=model)

    '''
    if data_args.dataset_name in ["c4", "c4-wiki-large"] and max_eval_samples > 0:
        eval_dataset_dict = {
            "c4": eval_dataset,
            
            "wikitext": lm_datasets["wikitext"] # .select(range(max_eval_samples))
            }
        misc_utils.swarn(f"Using evaluation data: {eval_dataset_dict}")
    else:
        eval_dataset_dict = None # eval_dataset
    '''
    # Initialize our Trainer
    print('##########################################')
    print(model)
    print('##########################################')
    # breakpoint()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None, # eval_dataset_dict if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None, # compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=None, # preprocess_logits_for_metrics
        #if training_args.do_eval and not is_torch_tpu_available()
        #else None,
        callbacks=[callback_utils.SaveFullModelCallback],
    )

    if return_trainer is True:
        return trainer

    # Training
    print(output_dir)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            print('we resume from checkpoint', checkpoint)
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, 'full_model.pth'))



def _evaluation_post_processing(
    prefix: str,
    metrics: Dict[str, Any],
    eval_dataset: datasets.Dataset,
) -> None:

    metrics[f"eval_{prefix}samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics[f"eval_{prefix}loss"])
    except OverflowError:
        perplexity = float("inf")

    metrics[f"{prefix}perplexity"] = perplexity


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
