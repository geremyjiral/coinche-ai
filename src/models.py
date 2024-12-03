from enum import Enum

from pydantic import BaseModel, Field


class Suit(str, Enum):
    HEARTS = "coeur"
    DIAMONDS = "carreau"
    CLUBS = "trèfle"
    SPADES = "pique"


class CardName(str, Enum):
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "valet"
    QUEEN = "dame"
    KING = "roi"
    ACE = "as"


class Card(BaseModel):
    suit: Suit
    value: int
    value_atout: int
    name: CardName
    order: int
    order_atout: int
    is_belote: bool = False

    def points(self, atout: Suit) -> float:
        if self.suit == atout:
            return self.value_atout
        return self.value

    def __hash__(self):
        return hash(
            (self.suit, self.value, self.value_atout, self.name, self.is_belote)
        )

    def __str__(self) -> str:
        return f"{self.name.value} de {self.suit.value}"

    def __repr__(self) -> str:
        return f"{self.name.value} de {self.suit.value}"


class Player(BaseModel):
    id: int
    name: str
    hand: list[Card] = Field(default_factory=list)
    team: int


class GameStage(str, Enum):
    BID = "bid"
    GAME = "game"


class Bid(BaseModel):
    player: int
    points: int | None
    suit: Suit | None
    is_coinche: bool = False
    is_pass: bool = False
    is_surcoinche: bool = False


class LogGame(BaseModel):
    bid: Bid
    attack_points: int
    defense_points: int
