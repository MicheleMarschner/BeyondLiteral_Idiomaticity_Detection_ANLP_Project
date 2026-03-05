# utils/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_submodule.utils.helper import save_plot


def plot_loss_curves(loss_curves: dict, save_dir: str | None = None) -> None:
    """
    Plot train loss (dense) and dev loss (sparse) over steps from saved loss_curves dict.
    """
    train_steps = loss_curves.get("train", {}).get("step", [])
    train_loss  = loss_curves.get("train", {}).get("loss", [])
    dev_steps   = loss_curves.get("dev", {}).get("step", [])
    dev_loss    = loss_curves.get("dev", {}).get("loss", [])

    fig, ax = plt.subplots(figsize=(7, 4))

    if train_steps and train_loss:
        ax.plot(train_steps, train_loss, label="train loss")
    if dev_steps and dev_loss:
        ax.plot(dev_steps, dev_loss, marker="o", markersize=4, linewidth=1, label="dev loss")

    best_idx = min(range(len(dev_loss)), key=lambda i: dev_loss[i])
    best_step = dev_steps[best_idx]
    best_val = dev_loss[best_idx]

    ax.axvline(best_step, linestyle="--", linewidth=1, color="red")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Dev Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_plot(fig, save_dir/"train_dev_loss_baseline.png")

