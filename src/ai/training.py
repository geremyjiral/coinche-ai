import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models import Bid, Card
from ai.models import CoincheAgent
from ai.utils import Experience, ReplayBuffer


class CoincheTrainer:
    def __init__(
        self,
        agent: CoincheAgent,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 32,
    ):
        self.agent = agent
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size

        # Initialize optimizers with proper parameters
        self.encoder_optimizer = Adam(
            agent.state_encoder.parameters(), lr=learning_rate
        )
        self.bidding_optimizer = Adam(
            agent.bidding_network.parameters(), lr=learning_rate
        )
        self.card_optimizer = Adam(
            agent.card_play_network.parameters(), lr=learning_rate
        )

        # Learning rate schedulers
        self.encoder_scheduler = StepLR(
            self.encoder_optimizer, step_size=100, gamma=0.95
        )
        self.bidding_scheduler = StepLR(
            self.bidding_optimizer, step_size=100, gamma=0.95
        )
        self.card_scheduler = StepLR(self.card_optimizer, step_size=100, gamma=0.95)

    def update_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        experiences = self.replay_buffer.sample(self.batch_size)

        # Separate bidding and card play experiences
        bidding_exp = [e for e in experiences if isinstance(e.action, Bid)]
        card_exp = [e for e in experiences if isinstance(e.action, Card)]

        if bidding_exp:
            self._update_bidding_network(bidding_exp)
        if card_exp:
            self._update_card_network(card_exp)

    def _update_bidding_network(self, experiences: list[Experience]):
        self.agent.state_encoder.train()
        self.agent.bidding_network.train()

        # Prepare batch data
        states = torch.stack(
            [
                self.agent.encode_game_state(e.game, e.game.current_player)
                for e in experiences
            ]
        )
        next_states = torch.stack(
            [
                self.agent.encode_game_state(e.next_game, e.next_game.current_player)
                for e in experiences
            ]
        )
        actions = torch.tensor(
            [e.action.points or 0 for e in experiences if isinstance(e.action, Bid)],
            device=self.agent.device,
        )
        rewards = torch.tensor(
            [e.reward for e in experiences], device=self.agent.device
        )

        # Compute current Q values
        state_features = self.agent.state_encoder(states)
        current_q = self.agent.bidding_network(state_features)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_features = self.agent.state_encoder(next_states)
            next_q = self.agent.bidding_network(next_features)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + self.gamma * max_next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.encoder_optimizer.zero_grad()
        self.bidding_optimizer.zero_grad()
        loss.backward()  # type: ignore
        torch.nn.utils.clip_grad_norm_(self.agent.state_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.agent.bidding_network.parameters(), 1.0)
        self.encoder_optimizer.step()  # type: ignore
        self.bidding_optimizer.step()  # type: ignore

    def _update_card_network(self, experiences: list[Experience]):
        self.agent.state_encoder.train()
        self.agent.card_play_network.train()

        # Prepare batch data
        states = torch.stack(
            [
                self.agent.encode_game_state(e.game, e.game.current_player)
                for e in experiences
            ]
        )
        next_states = torch.stack(
            [
                self.agent.encode_game_state(e.next_game, e.next_game.current_player)
                for e in experiences
            ]
        )
        actions = torch.tensor(
            [
                self.agent.card_to_index(e.action)
                for e in experiences  # type: ignore
                if isinstance(e.action, Card)
            ],
            device=self.agent.device,
        )
        rewards = torch.tensor(
            [e.reward for e in experiences], device=self.agent.device
        )

        # Compute current Q values
        state_features = self.agent.state_encoder(states)
        current_q = self.agent.card_play_network(state_features)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_features = self.agent.state_encoder(next_states)
            next_q = self.agent.card_play_network(next_features)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + self.gamma * max_next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.encoder_optimizer.zero_grad()
        self.card_optimizer.zero_grad()
        loss.backward()  # type: ignore
        torch.nn.utils.clip_grad_norm_(self.agent.state_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.agent.card_play_network.parameters(), 1.0)
        self.encoder_optimizer.step()  # type: ignore
        self.card_optimizer.step()  # type: ignore

    def step_schedulers(self):
        self.encoder_scheduler.step()
        self.bidding_scheduler.step()
        self.card_scheduler.step()
