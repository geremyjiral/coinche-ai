import random
from collections import deque
from typing import Deque

import torch
import torch.nn.functional as F

from game_rules import GameRules

from models import Bid, Card
from game import CoincheGame
from ai.models import CoincheAgent


class Experience:
    def __init__(
        self,
        game: CoincheGame,
        action: Card | Bid,
        reward: float,
        next_game: CoincheGame,
    ):
        self.game = game
        self.action = action
        self.reward = reward
        self.next_game = next_game


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class CoincheTrainer:
    def __init__(self, agent: CoincheAgent, learning_rate: float = 0.001):
        self.agent = agent
        self.replay_buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(
            list(agent.state_encoder.parameters())
            + list(agent.bidding_network.parameters())
            + list(agent.card_play_network.parameters()),
            lr=learning_rate,
        )

    def update_networks(self, batch_size: int = 32, gamma: float = 0.99):
        if len(self.replay_buffer) < batch_size:
            return

        experiences = self.replay_buffer.sample(batch_size)

        # Separate bidding and card play experiences
        bidding_exp = [e for e in experiences if isinstance(e.action, Bid)]
        card_exp = [e for e in experiences if isinstance(e.action, Card)]

        if bidding_exp:
            self._update_bidding_network(bidding_exp, gamma)
        if card_exp:
            self._update_card_network(card_exp, gamma)

    def _update_bidding_network(self, experiences: list[Experience], gamma: float):
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
            [e.action.points or 0 for e in experiences if isinstance(e.action, Bid)]
        )
        rewards = torch.tensor([e.reward for e in experiences])

        # Compute current Q values
        state_features = self.agent.state_encoder(states)
        current_q = self.agent.bidding_network(state_features)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_features = self.agent.state_encoder(next_states)
            next_q = self.agent.bidding_network(next_features)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + gamma * max_next_q

        # Compute loss and update
        loss: torch.Tensor = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore

    def _update_card_network(self, experiences: list[Experience], gamma: float):
        games = torch.stack(
            [
                self.agent.encode_game_state(e.game, e.game.current_player)
                for e in experiences
            ]
        )
        next_games = torch.stack(
            [
                self.agent.encode_game_state(e.next_game, e.next_game.current_player)
                for e in experiences
            ]
        )
        actions = torch.tensor(
            [
                self.agent.card_to_index(e.action)  # type: ignore
                for e in experiences
                if isinstance(e.action, Card)
            ]
        )
        rewards = torch.tensor([e.reward for e in experiences])

        # Compute current Q values
        state_features = self.agent.state_encoder(games)
        current_q = self.agent.card_play_network(state_features)
        current_q = current_q.gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_features = self.agent.state_encoder(next_games)
            next_q = self.agent.card_play_network(next_features)
            max_next_q = next_q.max(1)[0]
            target_q = rewards + gamma * max_next_q

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()  # type: ignore

    def compare_networks(self):
        # Compare state_encoder
        state_encoder_params_before = [
            p.clone() for p in self.agent.state_encoder.parameters()
        ]
        bidding_network_params_before = [
            p.clone() for p in self.agent.bidding_network.parameters()
        ]
        card_play_network_params_before = [
            p.clone() for p in self.agent.card_play_network.parameters()
        ]

        self.update_networks()

        state_encoder_params_after = [p for p in self.agent.state_encoder.parameters()]
        bidding_network_params_after = [
            p for p in self.agent.bidding_network.parameters()
        ]
        card_play_network_params_after = [
            p for p in self.agent.card_play_network.parameters()
        ]

        state_encoder_diff = [
            torch.sum(torch.abs(a - b)).item()
            for a, b in zip(state_encoder_params_before, state_encoder_params_after)
        ]
        bidding_network_diff = [
            torch.sum(torch.abs(a - b)).item()
            for a, b in zip(bidding_network_params_before, bidding_network_params_after)
        ]
        card_play_network_diff = [
            torch.sum(torch.abs(a - b)).item()
            for a, b in zip(
                card_play_network_params_before, card_play_network_params_after
            )
        ]

        print("State Encoder Differences:", state_encoder_diff)
        print("Bidding Network Differences:", bidding_network_diff)
        print("Card Play Network Differences:", card_play_network_diff)

    def compute_reward(self, game: CoincheGame, player_id: int) -> float:
        """Compute reward for the current state from player's perspective"""
        # Basic reward structure:
        # - Winning a trick: +1
        # - Taking successful contract: +10
        # - Failed contract: -10
        # - Points difference from contract: +/- 0.1 per point

        reward = 0.0

        # Reward for winning tricks
        if not game.atout or not game.current_bid or not game.current_bid.points:
            raise ValueError("Bid not set")
        if GameRules.determine_trick_winner(
            game.tricks[-1], game.atout, game.players
        ) in [
            game.players[player_id]
            for player_id in game.teams[game.players[player_id].team]
        ]:
            reward += sum(
                card.value if card.suit != game.atout else card.value_atout
                for card in game.tricks[-1]
            ) + (0 if len(game.tricks) < 8 else 10)
        else:
            reward -= sum(
                card.value if card.suit != game.atout else card.value_atout
                for card in game.tricks[-1]
            ) + (0 if len(game.tricks) < 8 else 10)
        if not len(game.tricks) == 8:
            return reward

        # Reward for contract
        points = sum(
            card.value if card.suit != game.atout else card.value_atout
            for trick in game.tricks
            for card in trick
            if GameRules.determine_trick_winner(trick, game.atout, game.players)
            in [
                game.players[player_id]
                for player_id in game.teams[game.players[player_id].team]
            ]
        ) + (
            10
            if GameRules.determine_trick_winner(
                game.tricks[-1], game.atout, game.players
            )
            in [
                game.players[player_id]
                for player_id in game.teams[game.players[player_id].team]
            ]
            else 0
        )
        is_player_attack = (
            game.current_bid.player in game.teams[game.players[player_id].team]
        )
        is_win = points >= game.current_bid.points
        reward += 3 * (
            (
                1
                if (is_player_attack and is_win)  # in attack team and win
                or (not is_player_attack and not is_win)  # or in defense team and lose
                else -1  # in attack team and lose or in defense team and win
            )
            * game.current_bid.points
            * (2 if game.current_bid.is_coinche else 1)
        )

        return reward
