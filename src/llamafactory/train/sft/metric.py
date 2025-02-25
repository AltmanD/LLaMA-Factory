# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
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
from typing import TYPE_CHECKING, Dict, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


def calculate_metrics(pred, ground_truth):
    # Initialize a dictionary to store metrics
    metrics = defaultdict(float)

    # Get the unique class labels
    classes = list(set(ground_truth))

    for cls in classes:
        # Create binary labels for the current class
        binary_ground_truth = [1 if x == cls else 0 for x in ground_truth]
        binary_pred = [1 if x == cls else 0 for x in pred]

        # Calculate metrics for the current class
        metrics[f"{cls}_acc"] = accuracy_score(binary_ground_truth, binary_pred)
        metrics[f"{cls}_precision"] = precision_score(binary_ground_truth, binary_pred, zero_division=0)
        metrics[f"{cls}_recall"] = recall_score(binary_ground_truth, binary_pred, zero_division=0)
        metrics[f"{cls}_f1"] = f1_score(binary_ground_truth, binary_pred, zero_division=0)

    return dict(metrics)


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"
    task1_classes = ['是', '否', '不说话']
    monitor_metrics = ['precision', 'recall', 'f1', 'acc']

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for metric in self.monitor_metrics:
            for cls in self.task1_classes:
                self.score_dict[f"{cls}_{metric}"] = []
        return result

    def __post_init__(self):
        self._dump()
    
    def _get_tasks_results(self, decoded_preds, decoded_labels, total_classes):
        pred_res, label_res = [], []
        for pred, label in zip(decoded_preds, decoded_labels):
            find = False
            for c in total_classes:
                if c in pred:
                    pred_res.append(c)
                    find = True
                    break
            if not find:
                pred_res.append("wrong")
            
            find = False
            for c in total_classes:
                if c in label:
                    label_res.append(c)
                    find = True
                    break
            if not find:
                label_res.append("wrong")
        return pred_res, label_res

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        task1_preds, task1_labels = self._get_tasks_results(decoded_preds, decoded_labels, self.task1_classes)
        task1_metrics = calculate_metrics(task1_preds, task1_labels)

        for metric in self.monitor_metrics:
            for cls in self.task1_classes:
                self.score_dict[f"{cls}_{metric}"].append(round(task1_metrics[f"{cls}_{metric}"] * 100, 4))

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
