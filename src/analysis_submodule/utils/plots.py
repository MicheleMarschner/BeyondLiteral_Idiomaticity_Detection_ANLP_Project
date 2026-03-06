from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_submodule.utils.helper import save_plot


def plot_loss_curves_nested(loss_curves: dict, save_dir: Path, model_family: str = "mBERT") -> None:
    """
    Plot train loss (dense) and dev loss (sparse) over steps from saved loss_curves dict.
    """
    train_steps = loss_curves.get("train", {}).get("step", [])
    train_loss = loss_curves.get("train", {}).get("loss", [])
    dev_steps = loss_curves.get("dev", {}).get("step", [])
    dev_loss = loss_curves.get("dev", {}).get("loss", [])

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    if train_steps and train_loss:
        sns.lineplot(x=train_steps, y=train_loss, ax=ax, label="train loss")

    if dev_steps and dev_loss:
        sns.lineplot(
            x=dev_steps,
            y=dev_loss,
            ax=ax,
            marker="o",
            markersize=4,
            linewidth=1,
            label="dev loss",
        )

        best_idx = min(range(len(dev_loss)), key=lambda i: dev_loss[i])
        best_step = dev_steps[best_idx]
        ax.axvline(best_step, linestyle="--", linewidth=1, color="red")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Dev Loss")
    ax.legend()

    save_path = Path(save_dir) / f"train_dev_loss_baseline_{model_family}.png"
    save_plot(fig, Path(save_path))


def plot_loss_curves_flat(loss_curves: dict, save_dir: Path, model_family: str = "logreg") -> None:
    """
    Plot train loss and dev loss from flat saved loss_curves dict
    """
    train_loss = loss_curves.get("train_loss", [])
    dev_loss = loss_curves.get("dev_loss", [])

    train_steps = list(range(1, len(train_loss) + 1))
    dev_steps = list(range(1, len(dev_loss) + 1))

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    if train_loss:
        sns.lineplot(x=train_steps, y=train_loss, ax=ax, label="train loss")

    if dev_loss:
        sns.lineplot(
            x=dev_steps,
            y=dev_loss,
            ax=ax,
            marker="o",
            markersize=4,
            linewidth=1,
            label="dev loss",
        )

        best_idx = min(range(len(dev_loss)), key=lambda i: dev_loss[i])
        best_step = dev_steps[best_idx]
        ax.axvline(best_step, linestyle="--", linewidth=1, color="red")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Dev Loss")
    ax.legend()

    save_path = Path(save_dir) / f"train_dev_loss_baseline_{model_family}.png"
    save_plot(fig, Path(save_path))


def plot_loss_curves_flat_comparison(
    loss_curves_1: dict,
    loss_curves_2: dict,
    label_1: str,
    label_2: str,
    save_dir: str | Path | None = None,
) -> None:
    """
    Plot two flat loss_curves dicts in one figure for comparison.
    """
    train_loss_1 = loss_curves_1.get("train_loss", [])
    dev_loss_1 = loss_curves_1.get("dev_loss", [])

    train_loss_2 = loss_curves_2.get("train_loss", [])
    dev_loss_2 = loss_curves_2.get("dev_loss", [])

    train_steps_1 = list(range(1, len(train_loss_1) + 1))
    dev_steps_1 = list(range(1, len(dev_loss_1) + 1))

    train_steps_2 = list(range(1, len(train_loss_2) + 1))
    dev_steps_2 = list(range(1, len(dev_loss_2) + 1))

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    color_1 = "tab:blue"
    color_2 = "tab:orange"

    if train_loss_1:
        sns.lineplot(
            x=train_steps_1,
            y=train_loss_1,
            ax=ax,
            color=color_1,
            label=f"{label_1} train loss",
        )

    if dev_loss_1:
        sns.lineplot(
            x=dev_steps_1,
            y=dev_loss_1,
            ax=ax,
            color=color_1,
            linewidth=1.5,
            linestyle="--",
            label=f"{label_1} dev loss",
        )
        best_idx_1 = min(range(len(dev_loss_1)), key=lambda i: dev_loss_1[i])
        best_step_1 = dev_steps_1[best_idx_1]
        ax.axvline(best_step_1, linestyle=":", linewidth=1, color="red")

    if train_loss_2:
        sns.lineplot(
            x=train_steps_2,
            y=train_loss_2,
            ax=ax,
            color=color_2,
            label=f"{label_2} train loss",
        )

    if dev_loss_2:
        sns.lineplot(
            x=dev_steps_2,
            y=dev_loss_2,
            ax=ax,
            linewidth=1.5,
            linestyle="--",
            color=color_2,
            label=f"{label_2} dev loss",
        )
        best_idx_2 = min(range(len(dev_loss_2)), key=lambda i: dev_loss_2[i])
        best_step_2 = dev_steps_2[best_idx_2]
        ax.axvline(best_step_2, linestyle=":", linewidth=1, color="red")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Dev Loss - Logreg")
    ax.legend()

    save_plot(fig, save_dir / "train_dev_loss_comparison_logreg.png")
    