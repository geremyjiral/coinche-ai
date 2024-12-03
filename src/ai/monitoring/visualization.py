import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns
from pathlib import Path

from .parameter_stats import LayerStats


class ParameterVisualizer:
    def __init__(self, save_dir: str = "src/ai/monitoring/plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_parameter_distributions(
        self, history: Dict[str, List[LayerStats]], save_path: str | None = None
    ):
        """Plot parameter value distributions over time for each layer."""
        num_layers = len(history)
        _, axes = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers))  # type: ignore
        if num_layers == 1:
            axes = [axes]

        for (name, layer_history), ax in zip(history.items(), axes):
            means = [stats.mean for stats in layer_history]
            stds = [stats.std for stats in layer_history]
            steps = range(len(layer_history))

            ax.plot(steps, means, label="Mean")
            ax.fill_between(
                steps,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.3,
            )
            ax.set_title(f"Parameter Distribution - {name}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Value")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(self.save_dir / save_path)  # type: ignore
        plt.close()

    def plot_gradient_flow(
        self, history: Dict[str, List[LayerStats]], save_path: str | None = None
    ):
        """Plot gradient magnitudes over time for each layer."""
        num_layers = len(history)
        _, axes = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers))  # type: ignore
        if num_layers == 1:
            axes = [axes]

        for (name, layer_history), ax in zip(history.items(), axes):
            grad_means = [
                stats.grad_mean
                for stats in layer_history
                if stats.grad_mean is not None
            ]
            grad_stds = [
                stats.grad_std for stats in layer_history if stats.grad_std is not None
            ]
            steps = range(len(grad_means))

            ax.plot(steps, grad_means, label="Gradient Mean")
            if grad_stds:
                ax.fill_between(
                    steps,
                    [m - s for m, s in zip(grad_means, grad_stds)],
                    [m + s for m, s in zip(grad_means, grad_stds)],
                    alpha=0.3,
                )
            ax.set_title(f"Gradient Flow - {name}")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Gradient Magnitude")
            ax.set_yscale("symlog")
            ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(self.save_dir / save_path)  # type: ignore
        plt.close()

    def plot_update_rates(
        self, history: Dict[str, List[LayerStats]], save_path: str | None = None
    ):
        """Plot parameter update rates over time."""
        plt.figure(figsize=(10, 6))  # type: ignore

        for name, layer_history in history.items():
            update_rates = [stats.update_rate for stats in layer_history]
            plt.plot(update_rates, label=name)  # type: ignore

        plt.title("Parameter Update Rates")  # type: ignore
        plt.xlabel("Training Step")  # type: ignore
        plt.ylabel("Update Rate")  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # type: ignore
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path)  # type: ignore
        plt.close()

    def plot_convergence_heatmap(
        self,
        history: Dict[str, List[LayerStats]],
        window_size: int = 100,
        save_path: str | None = None,
    ):
        """Plot convergence metrics as a heatmap."""
        metrics: Dict[str, Dict[str, float]] = {}
        for name, layer_history in history.items():
            if len(layer_history) < window_size:
                continue

            # Calculate stability metrics
            update_rates = np.array(
                [stats.update_rate for stats in layer_history[-window_size:]]
            )
            grad_means = np.array(
                [
                    stats.grad_mean
                    for stats in layer_history[-window_size:]
                    if stats.grad_mean is not None
                ]
            )

            metrics[name] = {
                "update_stability": float(np.std(update_rates)),
                "grad_stability": float(np.std(grad_means))
                if len(grad_means) > 0
                else 0,
            }

        if not metrics:
            return

        data = np.array(
            [[m["update_stability"], m["grad_stability"]] for m in metrics.values()]
        )
        plt.figure(figsize=(8, len(metrics) * 0.5 + 2))  # type: ignore
        sns.heatmap(  # type: ignore
            data,
            annot=True,
            fmt=".2e",
            xticklabels=["Update Stability", "Gradient Stability"],
            yticklabels=list(metrics.keys()),
            cmap="viridis",
        )
        plt.title("Convergence Metrics Heatmap")  # type: ignore
        plt.tight_layout()

        if save_path:
            plt.savefig(self.save_dir / save_path)  # type: ignore
        plt.close()
