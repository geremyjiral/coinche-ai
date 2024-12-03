from typing import Any
import torch
from pathlib import Path
from ai.models import CoincheAgent


class CheckpointManager:
    def __init__(self, save_dir: str = "src/ai/checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self, agent: CoincheAgent, episode: int, metrics: dict[str, Any] = {}
    ):
        checkpoint: dict[str, Any] = {
            "episode": episode,
            "state_encoder": agent.state_encoder.state_dict(),
            "bidding_network": agent.bidding_network.state_dict(),
            "card_play_network": agent.card_play_network.state_dict(),
            "metrics": metrics,
        }

        path = self.save_dir / f"checkpoint_ep{episode}.pt"
        torch.save(checkpoint, path)  # type: ignore

    def load_checkpoint(
        self, agent: CoincheAgent, episode: int | None = None
    ) -> dict[str, Any]:
        if episode is None:
            # Load latest checkpoint
            checkpoints = list(self.save_dir.glob("checkpoint_ep*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            path = max(checkpoints, key=lambda p: int(p.stem.split("ep")[1]))
        else:
            path = self.save_dir / f"checkpoint_ep{episode}.pt"

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {path} not found")

        checkpoint: dict[str, Any] = torch.load(path, weights_only=True)  # type: ignore
        agent.state_encoder.load_state_dict(checkpoint["state_encoder"])
        agent.bidding_network.load_state_dict(checkpoint["bidding_network"])
        agent.card_play_network.load_state_dict(checkpoint["card_play_network"])

        return checkpoint.get("metrics", {})
