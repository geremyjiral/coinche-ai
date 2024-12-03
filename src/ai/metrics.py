from dataclasses import dataclass
from typing import Any
import json
from pathlib import Path
import matplotlib.pyplot as plt


@dataclass
class EpisodeMetrics:
    episode: int
    episode_rewards: dict[int, float]
    total_reward: float
    win_rate: float
    average_points: float
    successful_contracts: int
    failed_contracts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "total_reward": self.total_reward,
            "win_rate": self.win_rate,
            "average_points": self.average_points,
            "successful_contracts": self.successful_contracts,
            "failed_contracts": self.failed_contracts,
            "episode_rewards": self.episode_rewards,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeMetrics":
        return cls(**data)


class MetricsTracker:
    def __init__(self, save_dir: str = "src/ai/training_metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics: list[EpisodeMetrics] = []

    def add_episode_metrics(self, metrics: EpisodeMetrics):
        self.metrics.append(metrics)

    def save_metrics(self, filename: str = "metrics.json"):
        path = self.save_dir / filename
        with open(path, "w") as f:
            json.dump([m.to_dict() for m in self.metrics], f)

    def load_metrics(self, filename: str = "metrics.json"):
        path = self.save_dir / filename
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                self.metrics = [EpisodeMetrics.from_dict(m) for m in data]

    def plot_metrics(self, save_path: str | None = None):
        episodes = [m.episode for m in self.metrics]
        rewards = [m.total_reward for m in self.metrics]
        win_rates = [m.win_rate for m in self.metrics]
        player_rewards: dict[int, list[float]] = {}
        for m in self.metrics:
            for player, reward in m.episode_rewards.items():
                if player not in player_rewards:
                    player_rewards[int(player)] = []
                player_rewards[int(player)].append(reward)

        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))  # type: ignore

        # Plot player rewards

        # Plot rewards
        ax1.plot(episodes, rewards, label="Total Reward")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Training Rewards over Time")
        ax1.legend()

        # Plot win rate
        ax2.plot(episodes, win_rates, label="Win Rate", color="green")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Win Rate over Time")
        ax2.legend()

        for player, rewards in player_rewards.items():
            ax3.plot(episodes, rewards, label=f"Player {player} Reward")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Reward")
        ax3.set_title("Player Rewards over Time")
        ax3.legend()

        plt.tight_layout()  # type: ignore
        if save_path:
            plt.savefig(save_path)  # type: ignore
        plt.close()  # type: ignore
