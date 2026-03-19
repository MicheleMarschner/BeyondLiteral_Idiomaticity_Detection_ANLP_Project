import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import ParameterGrid
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, PreTrainedModel

from typing import Dict, Tuple, Any, List

from utils.helper import set_seeds
from models.BERTs.param_grid import mBERT_grid, modernBERT_grid
from logger.wandb_logger import WandbDevCurveCallback
from models.BERTs.model import (
    tokenize_input,
    compute_metrics,
    freeze_encoder,
    compute_best_train_macro_f1,
)


def get_model_setup(model_family: str) -> Tuple[str, Dict[str, Any], bool]:
    """Resolve model identifier, parameter grid, and probe setting"""
    is_probe = model_family.endswith("_probe")
    base_family = model_family.replace("_probe", "")

    if base_family == "mBERT":
        model_id = "bert-base-multilingual-cased"
        param_grid = mBERT_grid
    elif base_family == "modernBERT":
        model_id = "answerdotai/ModernBERT-base"
        param_grid = modernBERT_grid
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    return model_id, param_grid, is_probe


def build_run_name(
    tokenization_config: Dict[str, Any],
    learning_config: Dict[str, Any],
    seed: int,
) -> str:
    """Build run name from tokenization and learning configuration"""
    return (
        f"ml{tokenization_config['max_length']}"
        f"__lr{learning_config['learning_rate']}"
        f"__ep{learning_config['num_train_epochs']}"
        f"__wd{learning_config['weight_decay']}"
        f"__bs{learning_config['batch_size']}"
        f"__seed{seed}"
    )


def build_training_args(
    run_dir: Path,
    run_name: str,
    learning_config: Dict[str, Any],
    seed: int,
) -> TrainingArguments:
    """Build HF training arguments"""
    return TrainingArguments(
        output_dir=str(run_dir),

        # Hyperparameters
        learning_rate=learning_config["learning_rate"],
        per_device_train_batch_size=learning_config["batch_size"],
        per_device_eval_batch_size=learning_config["batch_size"],
        num_train_epochs=learning_config["num_train_epochs"],
        weight_decay=learning_config["weight_decay"],

        # Evaluation & saving
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        save_only_model=True,      # avoids optimizer.pt/scheduler.pt
        load_best_model_at_end=True,
        metric_for_best_model="macro-F1",
        greater_is_better=True,

        # Reproducibility
        seed=seed,
        logging_dir=str(run_dir / "logs"),
        logging_steps=50,
        run_name=run_name,
        report_to="none",
    )


def extract_loss_curves(trainer: Trainer) -> Dict[str, Any]:
    """Extract the loss curves and F1 scores from trainer history"""
    # save the loss curves and f1 scores
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    dev_macro_f1 = []

    for row in trainer.state.log_history:
        if "loss" in row and "step" in row:
            train_steps.append(row["step"])
            train_loss.append(row["loss"])
        if "eval_loss" in row and "step" in row:
            eval_steps.append(row["step"])
            eval_loss.append(row["eval_loss"])
        if "eval_macro-F1" in row and "step" in row:
            dev_macro_f1.append(row["eval_macro-F1"])

    loss_curves = {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "dev_steps": eval_steps,
        "dev_loss": eval_loss,
        "dev_macro_f1": dev_macro_f1,
        "best_step": trainer.state.global_step,
    }
    return loss_curves


class BERTRunner:
    def prepare_features(self,
        params: Dict[str, Any],
        config: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """Fit a featurizer on train text and transform train/test into feature matrices"""

        set_seeds(config['seed'])
        train_dataset, dev_dataset, tokenizer = tokenize_input(params, train_df, test_df)

        return train_dataset, dev_dataset, tokenizer


    def initialize(self, params: Dict[str, Any], seed: int=51, model_family: str="mBERT") -> PreTrainedModel:
        """Load a binary sequence classifier from params and return the initialized model"""
        model = AutoModelForSequenceClassification.from_pretrained(
                    params.get("model_identifier"),
                    num_labels=2,
                    id2label={0: "Idiom", 1: "Literal"},
                    label2id={"Idiom": 0, "Literal": 1},
        )
        return model


    def tune(self,
        config: Dict[str, Any],
        model_path: Path,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        wandb_run: Any = None,
        threshold: float = 0.5
    ) -> Tuple[Trainer, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Grid-search hyperparameters, save the best model bundle, and return best model and other results"""

        out_dir = model_path / "training"
        results = []
        best_dev_macro_f1_overall = -1.0
        best_curves = None
        best_model = None
        best_params = None
        best_tokenizer = None

        model_family = config["model_family"]
        model_id, param_grid, is_probe = get_model_setup(model_family)

        tok_grid = list(ParameterGrid(param_grid["tok_space"]))
        learning_grid = list(ParameterGrid(param_grid["learning_space"]))

        # Run through hyperparameter grid
        for tokenization_config in tok_grid:
            train_data, dev_data, tokenizer = self.prepare_features(
                params={**tokenization_config, "model_identifier": model_id},
                config=config,
                train_df=train_df,
                test_df=dev_df
            )

            for learning_config in learning_grid:
                set_seeds(config['seed'])
                model = self.initialize(params={**learning_config, "model_identifier": model_id})
                model.resize_token_embeddings(len(tokenizer))

                if is_probe:
                    model = freeze_encoder(model)

                run_name = build_run_name(
                    tokenization_config=tokenization_config,
                    learning_config=learning_config,
                    seed=config["seed"],
                )
                run_dir = out_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                training_args = build_training_args(
                    run_dir=run_dir,
                    run_name=run_name,
                    learning_config=learning_config,
                    seed=config["seed"],
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_data,
                    eval_dataset=dev_data,
                    compute_metrics=compute_metrics,
                    callbacks=[WandbDevCurveCallback(wandb_run=wandb_run)],
                )

                trainer.train()
                best_dev_macro_f1 = float(trainer.state.best_metric)

                # compute train macro f1 score
                best_train_macro_f1 = compute_best_train_macro_f1(trainer, train_data)

                loss_curves = extract_loss_curves(trainer=trainer)

                results.append({
                    **tokenization_config,
                    **learning_config,
                    "best_dev_macro_f1": best_dev_macro_f1,
                    "best_train_macro_f1": float(best_train_macro_f1),
                })

                if best_dev_macro_f1 > best_dev_macro_f1_overall:
                    best_dev_macro_f1_overall = best_dev_macro_f1
                    best_model = trainer
                    best_params = {**tokenization_config, **learning_config, "model_identifier": model_id} # add model identifier, required during evaluation.
                    best_tokenizer = tokenizer
                    best_curves = loss_curves

        if best_model is None:
            raise RuntimeError("Tuning failed: no valid parameter combination produced a trained model.")

        best_model_dir = model_path
        best_model.model.save_pretrained(best_model_dir, safe_serialization=True)
        best_tokenizer.save_pretrained(best_model_dir)

        return best_model, results, best_params, best_curves


    def predict_proba(self, trainer, X: Any) -> np.ndarray:
        """Wrapper to get literal-class probabilities from a model"""
        
        out = trainer.predict(X)
        logits = out.predictions  

        # softmax -> probabilities
        logits = np.asarray(logits)
        logits = logits - logits.max(axis=1, keepdims=True)  
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        proba_literal = probs[:, 1]   # class 1
        
        return proba_literal.astype(float)