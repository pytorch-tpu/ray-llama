# Copyright 2023 The Google Cloud ML Accelerators Authors. All Rights Reserved.
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
#
#
# This code is adapted from
#
# huggingface/transformers:
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# ==============================================================================
"""
Utility functions for training Llama from HuggingFace
with PyTorch/XLA SPMD.

This is adapted from `run_clm.py` with minimal changes. This only
serves to show how you might adapt a Huggingface model to run on Ray.

"""
import os

from dataclasses import dataclass, field
import math
import transformers
from itertools import chain
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
#from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import logging

import datasets
from datasets import load_dataset, Dataset, DatasetDict

import json
import sys
from typing import Any, Mapping, Optional, Tuple

import evaluate

import torch

try:
    import torch_xla
    import torch_xla.debug.profiler as xp
    import torch_xla.core.xla_model as xm
    import torch_xla.experimental.xla_sharding as xs
    import torch_xla.runtime as xr
    xr.use_spmd()

    from torch_xla.distributed.fsdp import checkpoint_module
    from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear
    print("Successfully imported PyTorch/XLA.")
except ImportError as e:
    print("Failed to load PyTorch/XLA.")
    print("This is expected to fail on a CPU.")


check_min_version("4.32.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "The model type."}
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
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
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
    spmd_iota_mesh: bool = field(
        default=False,
        metadata={
            "help": (
                "Use the iota mesh instead of HybridMesh",
            )
        },
    )
    spmd_dcn_parallelism: int = field(
        default=1,
        metadata={
            "help": (
                "Number of slices to run in data parallel"
            )
        },
    )
    spmd_grad_chkpt: bool = field(
        default=False,
        metadata={
            "help": (
                "Apply gradient checkpointing to the model"
            )
        },
    )
    spmd_fsdp_sharding: bool = field(
        default=False,
        metadata={
            "help": (
                "Will apply XLA SPMD to run FSDP"
            )
        },
    )
    spmd_batch_sharding: bool = field(
        default=False,
        metadata={
            "help": (
                "Will apply XLA SPMD to shard the input along the batch dimension"
            )
        },
    )
    spmd_tensor_sharding: int = field(
        default=0,
        metadata={
            "help": (
                "Will apply XLA SPMD to shard the weights along two dimensions (num_devices / spmd_tensor_sharding, spmd_tensor_sharding)"
            )
        },
    )
    spmd_2d_sharding: int = field(
        default=0,
        metadata={
            "help": (
                "Will apply XLA SPMD to 2D sharding, i.e., weights + activations, and spmd_2d_sharding specifies the model dimension"
            )
        },
    )
    spmd_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Will print debug information"
            )
        },
    )

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
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
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
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
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



def load_huggingface_args(model_name: str, args: Mapping[str, Any]) -> Tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    print("Loading HuggingFace Arguments.")
    args["config_name"] = os.path.join("configs", model_name + ".json")
    print("Using config: ", args["config_name"])
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args)
    training_args.spmd_batch_sharding = model_args.spmd_batch_sharding or model_args.spmd_fsdp_sharding
    training_args.spmd_fsdp_sharding = model_args.spmd_fsdp_sharding
    training_args.spmd_tensor_sharding = model_args.spmd_tensor_sharding
    training_args.spmd_2d_sharding = model_args.spmd_2d_sharding
    training_args.spmd_dcn_parallelism = model_args.spmd_dcn_parallelism
    training_args.spmd_iota_mesh = model_args.spmd_iota_mesh
    training_args.report_to = []
    return model_args, data_args, training_args


def configure_loggers(should_log: bool = True):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = logger.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_checkpoint(training_args: TrainingArguments) -> Optional[str]:
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
            message =  (f"Checkpoint detected, resuming training at {last_checkpoint}. "
                        "To avoid this behavior, change the --output_dir or add "
                        "`--overwrite_output_dir` to train from scratch.")
            logger.info(message)
            print(message)
    return last_checkpoint


def get_tokenizer(model_args: ModelArguments) -> AutoTokenizer:
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
    return tokenizer


def get_model(tokenizer: AutoTokenizer,
              model_args: ModelArguments,
              training_args: TrainingArguments) -> AutoModelForCausalLM:
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
 
    num_devices = xr.global_runtime_device_count()
    device_ids = torch.arange(num_devices)
    def get_mesh(ici_mesh_shape, dcn_mesh_shape=None, axis_names=None):
        #dcn_mesh_shape=None

        if model_args.spmd_iota_mesh:
            mesh_shape = ici_mesh_shape
            if dcn_mesh_shape is not None:
                assert len(ici_mesh_shape) == len(dcn_mesh_shape)
                mesh_shape = tuple(i * d for i, d in zip(ici_mesh_shape, dcn_mesh_shape))
            return xs.Mesh(device_ids, mesh_shape, axis_names)
        else:
            return xs.HybridMesh(ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape, axis_names=axis_names)

    # Pass the sharding parameters to the model config
    config.spmd_debug = model_args.spmd_debug
    config.spmd_fsdp_sharding = model_args.spmd_fsdp_sharding
    assert model_args.spmd_2d_sharding == 0 or model_args.spmd_tensor_sharding == 0, 'Only one of --spmd_2d_sharding or --spmd_tensor_sharding can be specified'
    model_axis = max(model_args.spmd_2d_sharding + model_args.spmd_tensor_sharding, 1)
    dcn_axis = model_args.spmd_dcn_parallelism
    data_axis = num_devices // model_axis // dcn_axis

    # Place DCN on an independent axis in the mesh. Model parameters should be
    # replicated along the DCN axis, and inputs and activations should have
    # the batch dimension sharded along the combined DCN and data axes.
    ici_mesh_shape = (1, data_axis, model_axis)
    dcn_mesh_shape = (dcn_axis, 1, 1)
    config.spmd_mesh = get_mesh(ici_mesh_shape=ici_mesh_shape, dcn_mesh_shape=dcn_mesh_shape, axis_names=('dcn', 'data', 'model'))
    training_args.spmd_mesh = config.spmd_mesh

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print('Using dtype', model_args.torch_dtype)
    model = model.to(xm.xla_device(), dtype=getattr(torch, model_args.torch_dtype))

    # Replace the linear layer
    model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

    for name, param in model.named_parameters():
        print(name, torch_xla._XLAC._get_xla_sharding_spec(param))

    if model_args.spmd_grad_chkpt:
        print("Applying gradient checkpointing")
        for i, block in enumerate(model.model.layers):
            # LLaMA-specific
            model.model.layers[i] = checkpoint_module(block)
    return model


def get_raw_datasets(data_args: DataTrainingArguments,
                     model_args: ModelArguments) -> DatasetDict:
    if data_args.dataset_name is not None:
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
    return raw_datasets


def preprocess_datasets(
        tokenizer: AutoTokenizer,
        raw_datasets: DatasetDict,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments) -> Tuple[Optional[Dataset], Optional[Dataset]]:
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
        #with CaptureLogger(tok_logger) as cl:
        #    output = tokenizer(examples[text_column_name])
        output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        """
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        """
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
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
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

    train_dataset = None
    eval_dataset = None
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
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    return train_dataset, eval_dataset


def initialize_trainer(
    model: AutoModelForCausalLM,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer) -> Trainer:
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    if training_args.do_eval:
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )


def train(training_args: TrainingArguments,
          data_args: DataTrainingArguments,
          trainer: Trainer,
          train_dataset: Dataset,
          last_checkpoint: Optional[str] = None,
          should_save: bool = False) -> Mapping[str, Any]:
    print("starting to train")

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    if should_save:
        trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    if data_args.max_train_samples is not None:
        max_train_samples = data_args.max_train_samples

    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return metrics


def eval(trainer: Trainer,
         data_args: DataTrainingArguments,
         eval_dataset: Dataset) -> Mapping[str, Any]:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    max_eval_samples = len(eval_dataset)
    if num_samples_override is not None:
        max_eval_samples = num_samples_override
    elif data_args.max_eval_samples is not None:
        max_eval_samples = num_samples_override

    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return metrics


def push_to_hub(data_args: DataTrainingArguments,
                model_args: ModelArguments,
                training_args: TrainingArguments,
                trainer: Trainer):
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def run(
        train: bool,
        eval: bool,
        train_sample_override: Optional[int] = None,
        eval_sample_override: Optional[int] = None):
    model_args, data_args, training_args = load_huggingface_args()
    configure_loggers(training_args=training_args)
    set_seed(training_args.seed)

    server = xp.start_server(9012)
    logger.info('Profiling server started: {str(server)}')

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    raw_datasets = get_raw_datasets(
        data_args=data_args, model_args=model_args)
    tokenizer = get_tokenizer(model_args=model_args)
    model = get_model(
        tokenizer=tokenizer,
        model_args=model_args,
        training_args=training_args)
    train_dataset, eval_dataset = preprocess_datasets(
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        data_args=data_args,
        training_args=training_args)
    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer)

    train_metrics = None
    eval_metrics = None
    if train:
        train_metrics = train(
            trainer=trainer,
            training_args=training_args,
            data_args=data_args,
            last_checkpoint=get_checkpoint(training_args=training_args),
            train_dataset=train_dataset,
            num_samples_override=train_sample_override)
    if eval:
        eval_metrics = eval(
            trainer=trainer,
            data_args=data_args,
            eval_dataset=eval_dataset,
            num_samples_override=eval_sample_override)
    if push_to_hub:
        push_to_hub(data_args=data_args,
                    model_args=model_args,
                    training_args=training_args,
                    trainer=training_args)
    return train_metrics, eval_metrics


def init_hf(
    tokenizer_name: str = "gpt2",
    dataset_name: str = "wikitext",
    dataset_config_name: str = "wikitext-2-raw-v1",
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    num_train_epochs: int = 1,
    do_train: bool = True,
    cache_dir: str = "",
    output_dir: str = "/tmp/output",
    overwrite_output_dir: bool = True,
    model_name: str = "llama-",
    save_strategy: str = "no",
    logging_strategy: str = "no",
    spmd_2d_sharding: int = 1,
    torch_dtype: str = "bfloat16",
    dataloader_drop_last: bool = True,
    spmd_grad_chkpt: bool = True,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None) -> Mapping[str, Any]:
    """Initializes all Huggingface objects and returns as a dictionary."""

    input_args = dict(
          tokenizer_name=tokenizer_name,
          dataset_name=dataset_name,
          dataset_config_name=dataset_config_name,
          per_device_train_batch_size=per_device_train_batch_size,
          per_device_eval_batch_size=per_device_eval_batch_size,
          num_train_epochs=num_train_epochs,
          cache_dir=cache_dir,
          do_train=do_train,
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          save_strategy=save_strategy,
          logging_strategy=logging_strategy,
          spmd_2d_sharding=spmd_2d_sharding,
          torch_dtype=torch_dtype,
          dataloader_drop_last=dataloader_drop_last,
          spmd_grad_chkpt=spmd_grad_chkpt,
          max_train_samples=max_train_samples,
          max_eval_samples=max_eval_samples)

    if cache_dir is not None:
        try:
            os.mkdir(cache_dir)
        except:
            pass
    model_args, data_args, training_args = load_huggingface_args(model_name, input_args)
    set_seed(training_args.seed)

    raw_datasets = get_raw_datasets(
        data_args=data_args, model_args=model_args)
    print("Raw datasets loaded.")
    tokenizer = get_tokenizer(model_args=model_args)
    model = get_model(
        tokenizer=tokenizer,
        model_args=model_args,
        training_args=training_args)
    print("Tokenizer loaded.")
    train_dataset, eval_dataset = preprocess_datasets(
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        data_args=data_args,
        training_args=training_args)
    print("Datasets preprocessed.")
    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer)
    print("Trainer initialized.")

    return dict(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer=trainer)