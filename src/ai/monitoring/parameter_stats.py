import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path


@dataclass
class LayerStats:
    mean: float
    std: float
    min: float
    max: float
    grad_mean: float | None
    grad_std: float | None
    update_rate: float


class ParameterStats:
    def __init__(self, save_dir: str = "src/ai/monitoring/stats"):
        self.history: Dict[str, List[LayerStats]] = defaultdict(list)
        self.prev_params: Dict[str, torch.Tensor] = {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_layer_stats(
        self, name: str, param: torch.Tensor, update_window: int = 10
    ) -> LayerStats:
        with torch.no_grad():
            param_np: np.ndarray[Any, np.dtype[np.float32]] = (
                param.detach().cpu().numpy()  # type: ignore
            )
            grad_np = (  # type: ignore
                param.grad.detach().cpu().numpy() if param.grad is not None else None  # type: ignore
            )

            # Calculate update rate using previous parameters
            update_rate: float = 0.0
            if name in self.prev_params:
                param_diff = torch.norm(param - self.prev_params[name])  # type: ignore
                param_magnitude = torch.norm(self.prev_params[name])  # type: ignore
                update_rate = float(
                    (param_diff / param_magnitude).item()  # type: ignore
                    if param_magnitude > 0
                    else 0.0
                )

            self.prev_params[name] = param.detach().clone()

            return LayerStats(
                mean=float(np.mean(param_np)),
                std=float(np.std(param_np)),
                min=float(np.min(param_np)),
                max=float(np.max(param_np)),
                grad_mean=float(np.mean(grad_np)) if grad_np is not None else None,  # type: ignore
                grad_std=float(np.std(grad_np)) if grad_np is not None else None,  # type: ignore
                update_rate=update_rate,
            )

    def update_stats(self, model: torch.nn.Module) -> Dict[str, LayerStats]:
        current_stats: dict[str, LayerStats] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                stats = self.compute_layer_stats(name, param)
                self.history[name].append(stats)
                current_stats[name] = stats
        return current_stats

    def get_convergence_metrics(self, window_size: int = 100) -> Dict[str, float]:
        metrics: dict[str, float] = {}
        for name, history in self.history.items():
            if len(history) < window_size:
                continue

            # Calculate parameter stability
            recent_updates = [h.update_rate for h in history[-window_size:]]
            metrics[f"{name}_stability"] = float(np.std(recent_updates))

            # Calculate gradient trend
            recent_grads = [
                h.grad_mean for h in history[-window_size:] if h.grad_mean is not None
            ]
            if recent_grads:
                metrics[f"{name}_grad_trend"] = float(np.mean(recent_grads))

        return metrics

    def detect_anomalies(
        self, threshold: float = 3.0
    ) -> dict[str, list[dict[str, Any]]]:
        anomalies: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name, history in self.history.items():
            if len(history) < 2:
                continue

            # Check for unusual parameter changes
            update_rates = [h.update_rate for h in history]
            mean_rate = np.mean(update_rates)
            std_rate = np.std(update_rates)

            for i, rate in enumerate(update_rates):
                if abs(rate - mean_rate) > threshold * std_rate:
                    anomalies[name].append(
                        {
                            "type": "unusual_update",
                            "step": i,
                            "value": rate,
                            "mean": mean_rate,
                            "std": std_rate,
                        }
                    )

            # Check for vanishing/exploding gradients
            grads = [h.grad_mean for h in history if h.grad_mean is not None]
            if grads:
                grad_mean = np.mean(grads)
                # grad_std = np.std(grads)

                if abs(grad_mean) < 1e-7:
                    anomalies[name].append(
                        {"type": "vanishing_gradient", "mean": grad_mean}
                    )
                elif abs(grad_mean) > 1e3:
                    anomalies[name].append(
                        {"type": "exploding_gradient", "mean": grad_mean}
                    )

        return dict(anomalies)
