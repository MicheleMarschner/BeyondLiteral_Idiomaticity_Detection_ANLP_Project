
from typing import Dict, Tuple, Any, List
import numpy as np
from sklearn.model_selection import ParameterGrid
from pathlib import Path
import pandas as pd
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, Trainer, TrainingArguments)
from sklearn.metrics import f1_score
from datasets import Dataset

from utils.helper import set_seeds


def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenize text with padding and truncation."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def tokenize_input(params, train_data, dev_data):
    train_dataset = Dataset.from_pandas(train_data)
    dev_dataset = Dataset.from_pandas(dev_data)

    MODEL_NAME = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    # rename Label -> labels (as HF expects 'labels' -> we should do that already before when creating the data)
    if "Label" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("Label", "labels")
    if "Label" in dev_dataset.column_names:
        dev_dataset = dev_dataset.rename_column("Label", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    dev_dataset.set_format(type="torch", columns=cols)

    return train_dataset, dev_dataset, tokenizer


def compute_metrics(eval_pred):
    """Calculate Macro F1 score (average of F1 for each class)"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"macro-F1": macro_f1}


class mBERTRunner:
    def prepare_features(self, 
        params: Dict[str, Any], 
        exp_config: Dict[str, Any], 
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit a featurizer on train text and transform train/test into feature matrices"""
        
        set_seeds(exp_config['seed'])
        train_dataset, dev_dataset, tokenizer = tokenize_input(params, train_df, test_df)
        
        return train_dataset, dev_dataset, tokenizer

    
    def initialize(self, params: Dict[str, Any], seed: int=51, model_family: str="mBERT") -> PreTrainedModel:
        
        model = AutoModelForSequenceClassification.from_pretrained(
                    params.get("model_identifier"),
                    num_labels=2,
                    id2label={0: "Idiom", 1: "Literal"},
                    label2id={"Idiom": 0, "Literal": 1},
        )
        return model
    
    
    def tune(self, 
        exp_config: Dict[str, Any],
        model_path: Path,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        threshold: float=0.5
    ) -> Tuple[PreTrainedModel, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Grid-search hyperparameters, save the best model bundle, and return best model and other results"""
        
        out_dir = model_path / "training"
        results = []
        best_f1 = -1.0
        best_curves = None
        best_model = None
        best_params = None
        best_tokenizer = None

        tok_space = {
            "max_length": [256],
        }
        tok_grid = list(ParameterGrid(tok_space))


        learning_space = {
            "learning_rate": [2e-5],
            "num_train_epochs": [5],
            "weight_decay": [0.01],
        }
        learning_grid = list(ParameterGrid(learning_space))
        MODEL_ID = "bert-base-multilingual-cased"

        # Run through hyperparameter grid
        for tokenization_config in tok_grid:
            train_data, dev_data, tokenizer = self.prepare_features(params=tokenization_config, exp_config=exp_config, train_df=train_df, test_df=dev_df)

            for learning_config in learning_grid:
                set_seeds(exp_config['seed'])
                model = self.initialize(params={**learning_config, "model_identifier": MODEL_ID})
                
                run_name = (
                    f"ml{tokenization_config['max_length']}"
                    f"__lr{learning_config['learning_rate']}"
                    f"__ep{learning_config['num_train_epochs']}"
                    f"__wd{learning_config.get('weight_decay', 0.0)}"
                    f"__seed{exp_config['seed']}"
                )
                run_dir = out_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                training_args = TrainingArguments(
                    output_dir=str(run_dir),

                    # Hyperparameters (Baseline replication hence fixed)
                    learning_rate=learning_config["learning_rate"],
                    per_device_train_batch_size=32,
                    per_device_eval_batch_size=32,
                    num_train_epochs=learning_config["num_train_epochs"],
                    weight_decay=learning_config["weight_decay"],

                    # Evaluation & saving
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=2,
                    load_best_model_at_end=True,
                    metric_for_best_model="macro-F1",

                    # Reproducibility
                    seed=exp_config['seed'],
                    logging_dir=str(run_dir / "logs"),
                    logging_steps=50,
                    report_to=["none"]
                )

                trainer = Trainer(
                    model=model,          
                    args=training_args,                 
                    train_dataset=train_data,
                    eval_dataset=dev_data,
                    compute_metrics=compute_metrics
                )

                trainer.train()
                best_dev_f1 = float(trainer.state.best_metric) 

                # saving the loss curves
                train_steps, train_loss = [], []
                eval_steps, eval_loss = [], []

                for row in trainer.state.log_history:
                    if "loss" in row and "step" in row:
                        train_steps.append(row["step"])
                        train_loss.append(row["loss"])
                    if "eval_loss" in row and "step" in row:
                        eval_steps.append(row["step"])
                        eval_loss.append(row["eval_loss"])

                loss_curves = {
                    "train": {"step": train_steps, "loss": train_loss},
                    "dev":  {"step": eval_steps,  "loss": eval_loss},
                }

                results.append({
                    **tokenization_config,
                    **learning_config,
                    "best_dev_macro_f1": best_dev_f1,
                })

                if best_dev_f1 > best_f1:
                    best_f1 = best_dev_f1
                    best_model = trainer.model
                    best_params = {**tokenization_config, **learning_config}
                    best_tokenizer = tokenizer
                    best_curves = loss_curves
                
        if best_model is None:
            raise RuntimeError("Tuning failed: no devid parameter combination produced a trained model.")
                

        best_model_dir = Path(model_path / f"{exp_config['model_family']}")
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model.save_pretrained(best_model_dir, safe_serialization=True)
        best_tokenizer.save_pretrained(best_model_dir)

        return best_model, results, best_params, best_curves


    def predict_proba(self, model, X: Any) -> np.ndarray:
        """Wrapper to get positive-class probabilities from a model"""
        
        out = model.predict(X)
        logits = out.predictions  

        # softmax -> probabilities
        logits = np.asarray(logits)
        logits = logits - logits.max(axis=1, keepdims=True)  
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        proba_pos = probs[:, 1]   # class 1
        
        return proba_pos.astype(float)