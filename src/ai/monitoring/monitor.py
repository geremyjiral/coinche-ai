from typing import Dict, Any
import torch
import json
from pathlib import Path
from datetime import datetime

from .parameter_stats import ParameterStats
from .visualization import ParameterVisualizer


class NetworkMonitor:
    def __init__(
        self,
        save_dir: str = "src/ai/monitoring",
        visualization_interval: int = 1000,
        stats_interval: int = 100,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stats = ParameterStats()
        self.visualizer = ParameterVisualizer(str(self.save_dir / "plots"))
        self.visualization_interval = visualization_interval
        self.stats_interval = stats_interval
        self.step = 0

    def update(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Update monitoring statistics and generate visualizations."""
        self.step += 1
        current_stats = self.stats.update_stats(model)

        # Generate periodic statistics
        if self.step % self.stats_interval == 0:
            convergence_metrics = self.stats.get_convergence_metrics()
            anomalies = self.stats.detect_anomalies()

            stats_data: dict[str, Any] = {
                "step": self.step,
                "timestamp": datetime.now().isoformat(),
                "convergence_metrics": convergence_metrics,
                "anomalies": anomalies,
            }

            # Save statistics to file
            stats_file = self.save_dir / f"stats_{self.step}.json"
            with open(stats_file, "w") as f:
                json.dump(stats_data, f, indent=2)

        # Generate periodic visualizations
        if self.step % self.visualization_interval == 0:
            self.visualizer.plot_parameter_distributions(
                self.stats.history, f"param_dist_{self.step}.png"
            )
            self.visualizer.plot_gradient_flow(
                self.stats.history, f"grad_flow_{self.step}.png"
            )
            self.visualizer.plot_update_rates(
                self.stats.history, f"update_rates_{self.step}.png"
            )
            self.visualizer.plot_convergence_heatmap(
                self.stats.history, save_path=f"convergence_{self.step}.png"
            )

        return current_stats
