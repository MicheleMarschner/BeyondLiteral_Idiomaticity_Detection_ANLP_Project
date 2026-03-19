import numpy as np
from typing import Dict, Sequence, Tuple, Any

import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import PreTrainedTokenizerBase, AutoTokenizer, PreTrainedModel


def tokenize_function(
    examples: Dict[str, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize text with padding and truncation"""
    return tokenizer(
        examples["input"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def tokenize_input(
    params: Dict[str, Any],
    train_data: pd.DataFrame,
    dev_data: pd.DataFrame,
) -> Tuple[Dataset, Dataset, PreTrainedTokenizerBase]:
    """Tokenize train/dev data and return torch-formatted HF Datasets plus tokenizer"""
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    tokenizer = AutoTokenizer.from_pretrained(params["model_identifier"])
    tokenizer.add_special_tokens({"additional_special_tokens": ["<MWE>", "</MWE>"]})

    max_length = int(params["max_length"])

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    )
    dev_dataset = dev_dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    )

    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("label", "labels")
    if "label" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("label", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    dev_dataset.set_format(type="torch", columns=cols)

    return train_dataset, dev_dataset, tokenizer


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calculate Macro F1 score (average of F1 for each class)"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"macro-F1": macro_f1}


def freeze_encoder(model: PreTrainedModel) -> PreTrainedModel:
    """Freeze all layers except the default classification head"""
    for p in model.parameters():
        p.requires_grad = False

    # defreeze only head for training
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def compute_best_train_macro_f1(trainer: Any, train_data: Any) -> float:
    """Compute best train macro F1 score from the best loaded checkpoint"""
    train_metrics = trainer.predict(train_data)
    best_train_macro_f1 = compute_metrics(
        (train_metrics.predictions, train_metrics.label_ids)
    )["macro-F1"]
    return float(best_train_macro_f1)