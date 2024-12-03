import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product

from models import Bid, Card, Suit
from game import CoincheGame
from game_rules import GameRules


class CoincheStateEncoder(nn.Module):
    def __init__(
        self, input_dim: int = 32 * 3
    ):  # 32 cards * 3 (in hand, played, visible)
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return F.relu(self.fc3(x))


class BiddingNetwork(nn.Module):
    def __init__(self, state_dim: int = 64):
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        bids_possible = [
            Bid(player=id, points=points, suit=suit, is_coinche=is_coinche)
            for id, points, suit, is_coinche in product(
                range(4),
                list(range(80, 120, 10)) + list(range(115, 165, 5)),
                list(Suit),
                [True, False],
            )
        ]
        self.fc3 = nn.Linear(64, len(bids_possible))
        self.dropout = nn.Dropout(0.2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(state))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return F.softmax(self.fc3(x), dim=-1)


class CardPlayNetwork(nn.Module):
    def __init__(self, state_dim: int = 64):
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)  # One output per possible card
        self.dropout = nn.Dropout(0.2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(state))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        return F.softmax(self.fc3(x), dim=-1)


class CoincheAgent:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_encoder = CoincheStateEncoder().to(device)
        self.bidding_network = BiddingNetwork().to(device)
        self.card_play_network = CardPlayNetwork().to(device)

    def encode_game_state(self, game: CoincheGame, player_id: int) -> torch.Tensor:
        encoded = torch.zeros(32 * 3, device=self.device)

        # Encode cards in hand
        player = game.players[player_id]
        for card in player.hand:
            idx = self.card_to_index(card)
            encoded[idx] = 1

        # Encode cards played in current trick
        for card in game.current_trick:
            idx = self.card_to_index(card) + 32
            encoded[idx] = 1

        # Encode cards won
        for trick in game.tricks:
            for card in trick:
                idx = self.card_to_index(card) + 32 * 2
                encoded[idx] = 1

        return encoded.requires_grad_(True)

    def card_to_index(self, card: Card) -> int:
        suit_offset = {
            Suit.HEARTS: 0,
            Suit.DIAMONDS: 8,
            Suit.CLUBS: 16,
            Suit.SPADES: 24,
        }
        return suit_offset[card.suit] + card.order

    def select_bid(self, game: CoincheGame, player_id: int) -> Bid:
        self.state_encoder.eval()
        self.bidding_network.eval()

        with torch.no_grad():
            encoded_state = self.encode_game_state(game, player_id)
            state_features = self.state_encoder(encoded_state.unsqueeze(0))
            bid_probs = self.bidding_network(state_features).squeeze(0)

            # Get valid bids and their indices
            valid_bids = list(
                enumerate(
                    [
                        Bid(
                            player=player_id,
                            points=points,
                            suit=suit,
                            is_coinche=is_coinche,
                        )
                        for points, suit, is_coinche in product(
                            list(range(80, 120, 10)) + list(range(115, 165, 5)),
                            list(Suit),
                            [True, False],
                        )
                        if GameRules.is_valid_bid(
                            Bid(
                                player=player_id,
                                points=points,
                                suit=suit,
                                is_coinche=is_coinche,
                            ),
                            game.current_bid,
                        )
                    ]
                )
            )

            # Create a mask for valid bids
            mask = torch.zeros_like(bid_probs, dtype=torch.bool)
            for i, _ in valid_bids:
                mask[i] = True

            # Apply mask and select highest probability valid bid
            masked_probs = bid_probs.clone()
            masked_probs[~mask] = float("-inf")
            selected_bid_idx = torch.argmax(masked_probs).item()

        return next(
            (bid for i, bid in valid_bids if i == selected_bid_idx),
            Bid(player=player_id, is_pass=True, points=None, suit=None),
        )

    def select_card(self, game: CoincheGame, player_id: int) -> Card:
        self.state_encoder.eval()
        self.card_play_network.eval()

        with torch.no_grad():
            encoded_state = self.encode_game_state(game, player_id)
            state_features = self.state_encoder(encoded_state.unsqueeze(0))
            card_probs = self.card_play_network(state_features).squeeze(0)

            # Get valid cards that can be played
            player = game.players[player_id]
            valid_cards = [
                card
                for card in player.hand
                if game.current_bid
                and game.current_bid.suit
                and GameRules.is_play_valid(
                    card,
                    player,
                    game.current_trick,
                    game.current_bid.suit,
                    game.players,
                )
            ]
            if not valid_cards:
                raise ValueError("No valid cards to play")

            # Create a mask for valid cards
            mask = torch.zeros_like(card_probs, dtype=torch.bool)
            for card in valid_cards:
                mask[self.card_to_index(card)] = True

            # Apply mask and select highest probability valid card
            masked_probs = card_probs.clone()
            masked_probs[~mask] = float("-inf")
            selected_card_idx = torch.argmax(masked_probs).item()

            # Find the corresponding card
            for card in valid_cards:
                if self.card_to_index(card) == selected_card_idx:
                    return card

            # Fallback to first valid card if something goes wrong
            return valid_cards[0]
