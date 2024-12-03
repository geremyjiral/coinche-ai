from pydantic import BaseModel, ConfigDict

from game_rules import GameRules
from deck import create_deck, shuffle_deck, deal_cards
from logger import get_logger
from models import Card, Bid, LogGame, Suit, Player, GameStage

logger = get_logger(__name__)


class CoincheGame(BaseModel):
    players: list[Player] = []
    teams: list[list[int]] = [[], []]
    deck: list[Card] = create_deck()
    bids: list[Bid] = []
    current_bid: Bid | None = None
    tricks: list[list[Card]] = []
    current_trick: list[Card] = []
    current_player: int = 0
    phase: GameStage = GameStage.BID
    scores: list[int] = [0, 0]
    atout: Suit | None = None
    logs: list[LogGame] = []
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_player(self, player: Player):
        if len(self.teams[player.team]) < 2:
            self.players.append(player)
            self.teams[player.team].append(self.players.index(player))
        else:
            raise ValueError("Team already has 2 players")
        if len(self.players) == 4:
            self.players = [
                self.players[p]
                for p in [self.teams[j][i] for i in range(2) for j in range(2)]
            ]
            self.start_game()

    def start_game(self):
        if len(self.players) != 4:
            raise ValueError("Not enough players")
        if len(self.teams[0]) != 2 or len(self.teams[1]) != 2:
            raise ValueError("Teams not full")

        self.bids = []
        self.current_bid = None
        self.tricks = []
        self.current_trick = []
        self.current_player = 0
        self.atout = None
        self.deck = shuffle_deck(self.deck)
        hands = deal_cards(self.deck.copy())
        for player, hand in zip(self.players, hands):
            player.hand = hand
        self.phase = GameStage.BID

    def place_bid(self, bid: Bid):
        if GameRules.is_valid_bid(bid, self.current_bid):
            self.current_bid = bid
            self.bids.append(self.current_bid)
            self.current_player = (self.current_player + 1) % 4
        else:
            raise ValueError("Invalid bid")

    def pass_bid(self, player: Player):
        if GameRules.is_pass_valid(self.current_bid, self.players.index(player)):
            self.bids.append(
                Bid(
                    player=self.players.index(player),
                    is_pass=True,
                    points=None,
                    suit=None,
                )
            )
            self.current_player = (self.current_player + 1) % 4

            if len(self.bids) >= 3 and all(bid.is_pass for bid in self.bids[-3:]):
                logger.debug("End of bidding")
                self.end_bidding()
        else:
            raise ValueError("Invalid pass")

    def coinche(self, player: Player):
        if not self.current_bid:
            raise ValueError("No bid placed, you can't coinche")
        if GameRules.is_coinche_valid(
            self.current_bid, self.players.index(player), player, self.teams
        ):
            self.current_bid.is_coinche = True
            self.end_bidding()
        else:
            raise ValueError("Invalid coinche")

    def end_bidding(self):
        if not self.current_bid:
            if all(bid.is_pass for bid in self.bids[-4:]):
                self.new_game()
                return
            raise ValueError("No bid placed")
        self.atout = self.current_bid.suit
        self.current_player = 0
        self.phase = GameStage.GAME

    def play_card(self, player: Player, card: Card):
        if self.phase != GameStage.GAME:
            raise ValueError("Not in game phase")
        if player != self.get_current_player():
            raise ValueError("Not this player's turn")
        if not self.current_bid or not self.current_bid.suit:
            raise ValueError("Bid not valid")
        if GameRules.is_play_valid(
            card, player, self.current_trick, self.current_bid.suit, self.players
        ):
            self.current_trick.append(card)
            player.hand.remove(card)
            self.current_player = (self.current_player + 1) % 4
            if len(self.current_trick) == 4:
                self.end_trick()
        else:
            raise ValueError("Invalid play")

    def end_trick(self):
        if not self.current_bid:
            raise ValueError("No bid placed")
        if not self.current_bid.suit:
            raise ValueError("Bid not valid")
        winning_player = GameRules.determine_trick_winner(
            self.current_trick, self.current_bid.suit, self.players
        )
        self.tricks.append(self.current_trick)
        self.current_trick = []
        self.current_player = self.players.index(winning_player)
        if len(self.tricks) == 8:
            self.phase = GameStage.BID
            self.calculate_scores()

    def calculate_scores(self):
        if not self.current_bid:
            raise ValueError("No bid placed")
        if not self.current_bid.suit:
            raise ValueError("Bid not valid")
        if not self.current_bid.points:
            raise ValueError("Bid not valid")
        team_scores = [0, 0]
        for i, trick in enumerate(self.tricks):
            winning_player = GameRules.determine_trick_winner(
                trick, self.current_bid.suit, self.players
            )

            winning_team = winning_player.team
            team_scores[winning_team] += sum(
                card.value if card.suit != self.current_bid.suit else card.value_atout
                for card in trick
            )
            if i == 7:
                team_scores[winning_team] += 10
        bid_team = self.current_bid.player % 2
        defense_team = (bid_team + 1) % 2
        self.logs.append(
            LogGame(
                bid=self.current_bid,
                attack_points=team_scores[bid_team],
                defense_points=team_scores[defense_team],
            )
        )
        if team_scores[bid_team] < self.current_bid.points:
            logger.debug(
                f"Defense wins {self.current_bid.points=} {self.current_bid.is_coinche=}"
            )
            self.scores[bid_team] -= self.current_bid.points * (
                2 if self.current_bid.is_coinche else 1
            )
            self.scores[defense_team] += self.current_bid.points * (
                2 if self.current_bid.is_coinche else 1
            )
        else:
            logger.debug(
                f"Attack wins {self.current_bid.points=} {self.current_bid.is_coinche=}"
            )
            self.scores[bid_team] += self.current_bid.points * (
                2 if self.current_bid.is_coinche else 1
            )
            self.scores[defense_team] -= self.current_bid.points * (
                2 if self.current_bid.is_coinche else 1
            )

    def new_game(self):
        self.players = self.players[1:] + self.players[:1]
        self.start_game()

    def get_current_player(self) -> Player:
        return self.players[self.current_player]
