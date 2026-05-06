# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer

from evaluate import load
sari = load("sari")

import textstat
from bert_score import score
from tqdm import tqdm

# import torch.nn as nn
# import math

### ----------------- Helpers for dumping predicitons -------------------------- ### 
import os, json
from pathlib import Path

def _is_main_process() -> bool:
    # works for single-GPU, DDP, FSDP
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True

def _decode_preds_like_hf(tokenizer, predictions: np.ndarray) -> list[str]:
    # Handle either token ids or logits
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        # [bsz, seq, vocab] -> greedy ids
        predictions = predictions.argmax(-1)
    predictions = np.where(predictions != IGNORE_INDEX, predictions, tokenizer.pad_token_id)
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)

def _decode_labels(tokenizer, labels: np.ndarray) -> list[str]:
    labels = np.where(labels != IGNORE_INDEX, labels, tokenizer.pad_token_id)
    return tokenizer.batch_decode(labels, skip_special_tokens=True)

def _try_parse_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return s
##################################################################################


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()

@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            #result = self.score_dict
        self.score_dict = {"pred-tgt-dFKGL": [], "label-tgt-dFKGL": [], "SARI": [], "BERTScore_F1": []} # , "dFKGL_SARI": [] , "loss": [], "perplexity": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        # preds = eval_preds.predictions#[:, :-1, :].cpu().detach()
        # inputs = eval_preds.inputs#.cpu().detach()
        # labels = eval_preds.label_ids#[:, 1:].cpu().detach()

        # loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        # self.score_dict["loss"] = loss_fn(preds.view(-1, preds.size(-1)), labels.view(-1)  ).cpu().detach().item()
        # self.score_dict["perplexity"] = math.exp(self.score_dict["loss"])

        #preds = np.argmax(preds, axis=-1)

        # raw_inputs = np.where(numpify(eval_preds.inputs) != IGNORE_INDEX, numpify(eval_preds.inputs), self.tokenizer.pad_token_id)

        preds = numpify(eval_preds.predictions)
        labels = numpify(eval_preds.label_ids)
        inputs = numpify(eval_preds.inputs)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)
        inputs = np.where(inputs != IGNORE_INDEX, inputs, self.tokenizer.pad_token_id)
        
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

        sources = [source.split("\n")[3][:-9] for source in inputs] # remove the "assistant" on end of string
        grades = [int(source.split("\n")[2].split(" ")[-1].strip('.')) for source in inputs] # get the grade from the input prompt
        preds = [pred.removeprefix("assistant").removeprefix("\n").removeprefix("\n") for pred in preds] # remove the "assistant" at beginning of string

        # After decoding and extracting sources, preds, labels, compute metrics here
        self.score_dict["SARI"].append(sari.compute(sources=sources, predictions=preds, references=[[label] for label in labels])['sari'])
        bert_precision, bert_recall, bert_F1 = score(preds, sources, lang='en', verbose=True)
        self.score_dict["BERTScore_F1"].append(round(float(np.mean(bert_F1.numpy() * 100)), 2))

        # Compute FKGL and delta per grade group
        tgt_grade_deltas = []
        label_grade_deltas = []
        
        for pred, target_grade, label in tqdm(zip(preds, grades, labels)):
            # Compute FKGL for pred and label
            pred_fkgl = textstat.flesch_kincaid_grade(pred)
            label_fkgl = textstat.flesch_kincaid_grade(label)
            # Compute and append deltas
            tgt_grade_deltas.append(abs(pred_fkgl - target_grade))
            label_grade_deltas.append(abs(label_fkgl - target_grade))
        
        self.score_dict["pred-tgt-dFKGL"].append(np.mean(tgt_grade_deltas))
        self.score_dict["label-tgt-dFKGL"].append(np.mean(label_grade_deltas))

        # def compute_fkgl_x_sari(fkgl_delta, fkgl_alpha=0.5):
        #     sari_mean = np.mean(self.score_dict["SARI"])
        #     sari_beta = 1 - fkgl_alpha
        #     return 100 - sari_beta * (100 - sari_mean) - 10 * fkgl_alpha * fkgl_delta

        # self.score_dict["dFKGL_SARI"].append(compute_fkgl_x_sari(self.score_dict["pred-tgt-dFKGL"], fkgl_alpha=0.5))

        # NEW: one-shot JSONL dump at the end of eval, main process only
        if compute_result and _is_main_process():
            dump_path = os.getenv("LF_DUMP_JSONL")  # e.g., export LF_DUMP_JSONL=$OUTPUT_DIR/eval_predictions.jsonl
            if dump_path:
                out = Path(dump_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", encoding="utf-8") as f:
                    for i, p in enumerate(preds):
                        row = {
                            "id": i,
                            "tgt_grade": grades[i],
                            "source": sources[i],
                            "pred": _try_parse_json(p),
                            "label": labels[i][0],
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if compute_result:
            return self._dump()
