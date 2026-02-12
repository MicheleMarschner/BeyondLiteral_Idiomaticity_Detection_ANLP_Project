# Multilingual Idiomaticity Detection - Team: BeyondLiteral

This project implements the SemEval-2022 Task 2, Subtask A: Multilingual Idiomaticity Detection.



## Installation

The setup was tested on OS X. It uses docker and docker compose.

*Note:* For local development on Windows, use the Windows Subsystem for Linux (WSL) and follow the steps above for local development on Unix. Note that you are required to run Docker Desktop on the Host System and configure it for use with WSL: https://docs.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers

### Requirements
* Python (>= v)
* git
* Docker
* docker-compose 

#### 1. Clone the repository

```bash
git clone https://gitup.uni-potsdam.de/marschner5/beyondliteral_idiomaticity_detection_anlp_project.git to /your_desired_path/<myDeployment>
```
or if you are using ssh: 
```bash
git clone git@gitup.uni-potsdam.de:marschner5/beyondliteral_idiomaticity_detection_anlp_project.git
```

The repository has the following structure:
   ```text
    BeyondLiteral_Idiomaticity_Detection_ANLP_Project/
    ├── data/                           # Datasets (not tracked or partially tracked, depending on your setup)
    │   ├── raw/                        # Original/raw data dumps (as downloaded)
    │   ├── preprocessed/               # Cleaned/split/feature-ready CSVs (e.g., train/val/test)
    │   └── README.md                   # Notes on data sources + preprocessing decisions
    ├── documentation/                  # Project docs (reports, notes, figures, writeups)
    ├── experiments/                    # Per-run artifacts (one folder per experiment)
    │   ├── <run_name>/                 # Example folder: 
    │   │   ├── experiment_config.json  # Experiment config used for this run
    │   │   ├── best_params.json        # Best hyperparams from tuning
    │   │   ├── tuning_results.json     # Full grid search results
    │   │   ├── metrics.json            # Final evaluation metrics on test set
    │   │   ├── metrics.csv             # Final evaluation metrics on test set in tabular form (for analysis aggregation)
    │   │   ├── test_predictions.csv    # Predictions/probabilities per sample for test set
    │   │   └── *.joblib/*.pth          # Saved model weights
    ├── results/                        # Outputs from anaylsis
    │   ├── plots/                      
    │   └── tables/                     
    ├── src/                            # Source code
    │   ├── main.py                     # CLI entrypoint
    │   ├── config.py                   # Global settings, paths, constants
    │   ├── data.py                     # Data loading utilities (read CSVs, build splits, etc.)
    │   ├── training.py                 # High-Level Training orchestration (fit, tune, save artifacts)
    │   ├── evaluation.py               # Metrics + evaluation logic (macro-F1, slices, reports)
    │   ├── utils/                      # Generic helpers (I/O, seeding, path creation, logging)
    │   ├── experiments/                # Experiment definitions + grid expansion + runners
    │   │   └── experiments_template.py # ExperimentTemplate(s): settings, languages, models, seeds, etc.
    │   └── models/                     # Model implementations and wrappers
    │       ├── factory.py              # Maps model_family -> correct runner/model init
    │       ├── logreg/                 # Sklearn Implementation for comparison
    │       └── logreg_bare_metal/      # Bare-metal implementation
    ├── config.yaml                     
    ├── docker-compose.yaml             # Compose runner (mount data RO, persist experiments/results)
    ├── Dockerfile                      # Reproducible runtime image (installs deps, runs `src.main`)
    ├── pyproject.toml                  # Project metadata + dependencies (pip install -e .)
    ├── .dockerignore                   
    ├── .gitignore                      
    └── README.md                       # Main project documentation (how to run, reproduce, results)
   ```


## Running with Docker (recommended)

**Folder mounts**
- `./data` → `/app/data` (**read-only**)
- `./experiments` → `/app/experiments`
- `./results` → `/app/results`
- `./src/experiments/experiments_template.py` → `/app/src/experiments/experiments_template.py` (**read-only**)

To create a container for the first time or after code changes run:
```bash
mkdir -p experiments results
docker compose run --rm --build <service>
```

Run the container (without rebuild) after that with
```bash
docker compose run --rm <service>
```

# !TODO section

Available Services:
- `runner_train`: 
- `runner_eval`: 

```bash
docker compose run --rm train --overwrite
```


# Limitations
* Prompt "Tuning?": Delayed? for further projects

## ! TODO Missing


# Results

## ! TODO Missing