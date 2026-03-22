# Multilingual Idiomaticity Detection - Team: BeyondLiteral

This project implements the SemEval-2022 Task 2, Subtask A: Multilingual Idiomaticity Detection. It's a binary classification task: given a multiword expression (MWE) in context, predict whether it is used idiomatically or literally. The dataset is multilingual (EN/PT/GL) and includes the target sentence plus surrounding context (previous/next sentence), allowing us to study how different context and additional signals influence idiomaticity detection.

---
## Installation

The setup was tested on macOS. It uses docker and docker compose.

*Note:* For local development on Windows, use the Windows Subsystem for Linux (WSL) and follow the steps above for local development on Unix. Note that you are required to run Docker Desktop on the Host System and configure it for use with WSL: https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers

### Requirements
* Python (>= v3.10)
* git
* Docker
* docker-compose 

### Clone the repository

```bash
git clone https://gitup.uni-potsdam.de/marschner5/beyondliteral_idiomaticity_detection_anlp_project.git to /your_desired_path/<myDeployment>
```
or if you are using ssh: 
```bash
git clone git@gitup.uni-potsdam.de:marschner5/beyondliteral_idiomaticity_detection_anlp_project.git
```

The repository has the following high-level structure:
   ```text
   BeyondLiteral_Idiomaticity_Detection_ANLP_Project/
   ├── data/
   │   ├── raw/                 # semEval dataset files (see Data section)
   │   └── preprocessed/        # zero_shot_splits/, one_shot_splits/
   ├── documentation/           # project docs (reports, notes, figures, writeups)
   ├── experiments/             # one folder per run with artifacts (configs, metrics, predictions, model)
   ├── results/                 # aggregated experiment results, tables and plots
   └── src/                     # code (entrypoint: main.py)
   └── README.md                # main project documentation (how to run, reproduce, results)
   ```

---
## Weights & Biases

The experiment runs are available in the public W&B project:

**[beyondliteral_idiomaticity_detection_anlp](https://wandb.ai/michele-marschner-1-university-of-potsdam/beyondliteral_idiomaticity_detection_anlp/)**

The project contains logged runs, configurations, metrics, and selected artifacts generated during the experiments. Model checkpoints are stored locally and are not uploaded to W&B.

### Optional W&B Logging

W&B logging is optional. The project can be run without a W&B account.  
If you want to enable experiment tracking with W&B, create a `.env` file based on the provided example and fill in your credentials.

```bash
cp .env.example .env
```

```env
WANDB_API_KEY=wandb_api_key
WANDB_ENTITY=wandb_username
WANDB_PROJECT=beyondliteral_idiomaticity_detection_anlp
```

If no valid W&B configuration is provided, the project will continue to run without W&B logging.

---

## Running with Docker (recommended)

We provide a Docker-based workflow to ensure reproducibility across machines.

### Folder mounts
The container uses the following mounts (paths inside the container may differ depending on your compose file):
- `./data` →  dataset files (raw + preprocessed splits)
- `./experiments` → one folder per run (configs, checkpoints, predictions)
- `./results` → aggregated analysis outputs (tables/plots)

### Services
We provide three services, one for each stage of the pipeline:
- `train`: runs the experiments defined in src/experiments/experiments_template.py and writes run folders to ./experiments/
- `eval`: aggregates all finished runs into a summary overview master_metrics_long.csv
- `analyse`: creates specific data slices for deeper anaylsis, collects info in slices_overview.csv and generates indivdiual analysis tables and plots under ./results/<teammember>


### Workflow

1. Create output folders once:
```bash
mkdir -p experiments results
```

2. Train (build container on first run or after dependency/code changes):
```bash
docker compose run --rm --build train
```

3. Re-run training without rebuild:
```bash
docker compose run --rm train
```

4. Aggregate metrics across all runs:
```bash
docker compose run --rm eval
```

5. Generate tables/plots:
```bash
docker compose run --rm analyse
```

*Overwrite flag*
`--overwrite` re-runs experiments even if an experiment folder already exists. Without it an error will be thrown if you try to overvwrite

**Note on eval and analyse**: 
When running `eval` or `analyse`, the container will automatically attempt to download the experiments archive if no local experiment folders are present.

If the automatic download fails, manually download the archive from [here](https://drive.google.com/file/d/1rwKbbOPYdG6T8DcFCTzoPhzoZVyy0JtL/view?usp=sharing), ensure it is available as a .tar.gz file, and extract it in the project root.

---

## Data
This project uses the official dataset for SemEval-2022 Task 2 (Multilingual Idiomaticity Detection). The repository already contains the dataset files needed to run the pipeline (see `data/raw/Data/`).

Expected and needed raw files in `data/raw/Data/`:
* train_zero_shot.csv
* train_one_shot.csv
* dev.csv
* dev_gold.csv
* eval.csv

The preprocessed splits are included in the repository for convenience, but they can also be re-generated using:
`data/preprocessed/data_preprocessing_splitting.py`

### Experiments Folder
For running the 

---

## Experiment Configuration

Experiments are defined in `src/experiments/experiments_template.py`.  
Each run stores its exact configuration in `experiments/<run_name>/experiment_config.json`.

### Setting (`setting`)
- `zero_shot` — test MWEs are unseen during training (type-level generalization)
- `one_shot` — selected MWE types appear exactly once in training (minimal supervision)

### Language mode (`language_mode`)
- `per_language` — train and evaluate within one language (e.g., EN → EN)
- `multilingual` — train jointly on multiple languages and evaluate per language
- `cross_lingual` — train on one language and evaluate on another (e.g., EN → PT)

### Context window (`context`)
Controls which sentence(s) are used as input:
- `target` — only the target sentence
- `previous_target_next` — previous + target + next sentence

### Include MWE segment (`include_mwe_segment`)
- `True` — prepend the MWE string to the input (e.g., `MWE [SEP] ...`)

### Input transform (`transform`)
Controls how the MWE is represented in the input:
- `none` — no special marking
- `highlight` — mark the MWE span with special tokens (`<MWE> … </MWE>`)

### Extra features (`features`)
Optional signals added to the input:
- `empty` — no extra features
- `ner` — inline named-entity markup
- `glosses` — gloss words appended to provide a literal meaning cue

### Model family (`model_family`)
We support multiple model families for comparison:
- `logreg_tfidf` — logistic regression on TF-IDF features
- `logreg_word2vec` — logistic regression with Word2Vec features
- `mBERT_probe` — linear probe baseline (encoder frozen, train classifier head)
- `mBERT` — fine-tuned multilingual BERT classifier
- `modernBERT` — fine-tuned ModernBERT classifier

### Seed (`seed`)
- `seed` controls reproducibility. We run each configuration with a fixed seed by default

---

## Outputs and artifacts

### Per experiment run (stored in `./experiments/<run_name>/`)
Each experiment creates one folder containing all artifacts needed to reproduce the run and analyse it later:

- `experiment_config.json`  Exact configuration used for the run (setting, language mode, context, features, seed, etc.)

- `best_params.json`  Best hyperparameters selected during tuning (if tuning is enabled for the model)

- `tuning_results.json`  Full hyperparameter search results (useful for debugging and reporting)

- `metrics.json`  Final evaluation metrics on the test set (macro-F1, accuracy, confusion counts)

- `metrics.csv`  Same metrics as a single-row table (easier to aggregate across many runs)

- `test_predictions.csv`  Per-example predictions on the test set, including:`id, label, pred, proba`

- `learning_curves.json`   Training and dev curves saved during training

- Model artifacts (depending on model family)  e.g. saved weights / sklearn dumps such as `*.safetensors`, `*.joblib`


### Aggregated results (stored in `./results/`)
The evaluation and analysis stages produce aggregated files that combine results across all run folders:

- `master_metrics_long.csv`  One row per run × `eval_language` with overall metrics (used for your main plots/tables)

- `slices_overview.csv`  One row per test instance × run with slice metadata (used for deeper linguistic analysis)

- `./results/<teammember>`  Final plots and tables for the report

---

## Limitations
- **Single-seed runs:** Most experiments were run with one fixed seed due to computational constraints.  
  Results should therefore be interpreted as trends rather than fully variance-estimated averages.

- **Limited hyperparameter tuning:** Hyperparameter tuning was kept lightweight to fit the project timeline  
  (a larger tuning budget would likely improve absolute performance).
