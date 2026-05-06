# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity
from .trainer import CustomSeq2SeqTrainer
import json
import os
from dataclasses import asdict
from datetime import datetime


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

def save_debug_state(payload, save_dir, filename_prefix):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{filename_prefix}_{timestamp}.json"
    path = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[debug] wrote debug state to {path}")
    return path

def dump_llamafactory_state_to_json(
    model,
    tokenizer,
    model_args,
    data_args,
    training_args,
    generating_args,
    save_dir,
    filename_prefix="llamafactory",
    example_prompt=None,
):
    # GeneratingArguments as dict
    gen_args_dict = asdict(generating_args)

    # Training arguments, only generation related fields
    training_gen = {
        "per_device_eval_batch_size": getattr(training_args, "per_device_eval_batch_size", None),
        "per_device_train_batch_size": getattr(training_args, "per_device_train_batch_size", None),
        "predict_with_generate": getattr(training_args, "predict_with_generate", None),
        "generation_max_length": getattr(training_args, "generation_max_length", None),
        "generation_num_beams": getattr(training_args, "generation_num_beams", None),
    }

    # Model generation config
    try:
        gen_config = model.generation_config.to_dict()
    except Exception:
        try:
            gen_config = dict(model.generation_config)
        except Exception:
            gen_config = str(model.generation_config)

    # Tokenizer info
    tok_info = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }

    # Model config
    try:
        model_cfg = model.config.to_dict()
    except Exception:
        model_cfg = str(model.config)

    payload = {
        "backend": "llamafactory",
        "model_name_or_path": getattr(model_args, "model_name_or_path", None),
        "adapter_name_or_path": getattr(model_args, "adapter_name_or_path", None),
        "merged_model": True,   # set manually if you know this run uses merged weights
        "data_args": asdict(data_args),
        "training_args_generation": training_gen,
        "generating_args": gen_args_dict,
        "model_generation_config": gen_config,
        "model_config": model_cfg,
        "tokenizer_info": tok_info,
        "example_prompt": example_prompt,
    }

    return save_debug_state(payload, save_dir, filename_prefix)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    if training_args.do_eval:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        tokenizer.padding_side = 'left' # padding to right (otherwise SFTTrainer shows warning)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # # change the padding tokenizer value
    # model.config.pad_token_id = tokenizer.pad_token_id # updating model config
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    #training_args.label_names = ["labels"]
    #training_args.can_return_loss = True
    #training_args.include_loss_for_metrics = True
    training_args.include_inputs_for_metrics = True
    training_args.include_for_metrics = ["inputs"]
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)


    ### ------------------ Force greedy Gen kwargs ------------------ ###
    gen_cfg = generating_args
    gen_cfg.do_sample = False
    gen_cfg.num_beams = 1
    gen_cfg.temperature = 0.0
    gen_cfg.top_p = 1.0
    gen_cfg.top_k = -1
    gen_cfg.repetition_penalty = 1.0
    gen_cfg.max_new_tokens = 1024
    gen_cfg.max_length = None  # let max_new_tokens control length
    # ------------------------------------------------------------ ###
    # Keyword arguments for `model.generate`
    gen_kwargs = gen_cfg.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    ### ------------------ Debug: dump state to json ------------------ ###
    example_prompt = None
    save_dir = "/nethome/wlacroix/LLaMA-Factory/experiments/logs/debug"
    dump_llamafactory_state_to_json(model,
                                    tokenizer,
                                    model_args,
                                    data_args,
                                    training_args,
                                    generating_args,
                                    save_dir,
                                    filename_prefix="LF",
                                    example_prompt=example_prompt)
    ### --------------------------------------------------------------- ###

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                    dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
