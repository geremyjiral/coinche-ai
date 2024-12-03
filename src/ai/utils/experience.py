from dataclasses import dataclass
from collections import deque
import random
from typing import Deque

from game import CoincheGame
from models import Card, Bid


@dataclass
class Experience:
    game: CoincheGame
    action: Card | Bid
    reward: float
    next_game: CoincheGame


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
