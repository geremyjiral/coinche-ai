import random
from models import Card, Suit, CardName


SEVEN_HEARTS = Card(
    name=CardName.SEVEN,
    suit=Suit.HEARTS,
    value=0,
    value_atout=0,
    order=0,
    order_atout=0,
)
EIGHT_HEARTS = Card(
    name=CardName.EIGHT,
    suit=Suit.HEARTS,
    value=0,
    value_atout=0,
    order=1,
    order_atout=1,
)
NINE_HEARTS = Card(
    name=CardName.NINE,
    suit=Suit.HEARTS,
    value=0,
    value_atout=14,
    order=2,
    order_atout=6,
)
TEN_HEARTS = Card(
    name=CardName.TEN,
    suit=Suit.HEARTS,
    value=10,
    value_atout=10,
    order=6,
    order_atout=4,
)
JACK_HEARTS = Card(
    name=CardName.JACK,
    suit=Suit.HEARTS,
    value=2,
    value_atout=20,
    order=3,
    order_atout=7,
)
QUEEN_HEARTS = Card(
    name=CardName.QUEEN,
    suit=Suit.HEARTS,
    value=3,
    value_atout=3,
    order=4,
    order_atout=2,
)
KING_HEARTS = Card(
    name=CardName.KING, suit=Suit.HEARTS, value=4, value_atout=4, order=5, order_atout=3
)
ACE_HEARTS = Card(
    name=CardName.ACE,
    suit=Suit.HEARTS,
    value=11,
    value_atout=11,
    order=7,
    order_atout=5,
)

SEVEN_SPADES = Card(
    name=CardName.SEVEN,
    suit=Suit.SPADES,
    value=0,
    value_atout=0,
    order=0,
    order_atout=0,
)
EIGHT_SPADES = Card(
    name=CardName.EIGHT,
    suit=Suit.SPADES,
    value=0,
    value_atout=0,
    order=1,
    order_atout=1,
)
NINE_SPADES = Card(
    name=CardName.NINE,
    suit=Suit.SPADES,
    value=0,
    value_atout=14,
    order=2,
    order_atout=6,
)
TEN_SPADES = Card(
    name=CardName.TEN,
    suit=Suit.SPADES,
    value=10,
    value_atout=10,
    order=6,
    order_atout=4,
)
JACK_SPADES = Card(
    name=CardName.JACK,
    suit=Suit.SPADES,
    value=2,
    value_atout=20,
    order=3,
    order_atout=7,
)
QUEEN_SPADES = Card(
    name=CardName.QUEEN,
    suit=Suit.SPADES,
    value=3,
    value_atout=3,
    order=4,
    order_atout=2,
)
KING_SPADES = Card(
    name=CardName.KING, suit=Suit.SPADES, value=4, value_atout=4, order=5, order_atout=3
)
ACE_SPADES = Card(
    name=CardName.ACE,
    suit=Suit.SPADES,
    value=11,
    value_atout=11,
    order=7,
    order_atout=5,
)

SEVEN_DIAMONDS = Card(
    name=CardName.SEVEN,
    suit=Suit.DIAMONDS,
    value=0,
    value_atout=0,
    order=0,
    order_atout=0,
)
EIGHT_DIAMONDS = Card(
    name=CardName.EIGHT,
    suit=Suit.DIAMONDS,
    value=0,
    value_atout=0,
    order=1,
    order_atout=1,
)
NINE_DIAMONDS = Card(
    name=CardName.NINE,
    suit=Suit.DIAMONDS,
    value=0,
    value_atout=14,
    order=2,
    order_atout=6,
)
TEN_DIAMONDS = Card(
    name=CardName.TEN,
    suit=Suit.DIAMONDS,
    value=10,
    value_atout=10,
    order=6,
    order_atout=4,
)
JACK_DIAMONDS = Card(
    name=CardName.JACK,
    suit=Suit.DIAMONDS,
    value=2,
    value_atout=20,
    order=3,
    order_atout=7,
)
QUEEN_DIAMONDS = Card(
    name=CardName.QUEEN,
    suit=Suit.DIAMONDS,
    value=3,
    value_atout=3,
    order=4,
    order_atout=2,
)
KING_DIAMONDS = Card(
    name=CardName.KING,
    suit=Suit.DIAMONDS,
    value=4,
    value_atout=4,
    order=5,
    order_atout=3,
)
ACE_DIAMONDS = Card(
    name=CardName.ACE,
    suit=Suit.DIAMONDS,
    value=11,
    value_atout=11,
    order=7,
    order_atout=5,
)

SEVEN_CLUBS = Card(
    name=CardName.SEVEN, suit=Suit.CLUBS, value=0, value_atout=0, order=0, order_atout=0
)
EIGHT_CLUBS = Card(
    name=CardName.EIGHT, suit=Suit.CLUBS, value=0, value_atout=0, order=1, order_atout=1
)
NINE_CLUBS = Card(
    name=CardName.NINE, suit=Suit.CLUBS, value=0, value_atout=14, order=2, order_atout=6
)
TEN_CLUBS = Card(
    name=CardName.TEN, suit=Suit.CLUBS, value=10, value_atout=10, order=6, order_atout=4
)
JACK_CLUBS = Card(
    name=CardName.JACK, suit=Suit.CLUBS, value=2, value_atout=20, order=3, order_atout=7
)
QUEEN_CLUBS = Card(
    name=CardName.QUEEN, suit=Suit.CLUBS, value=3, value_atout=3, order=4, order_atout=2
)
KING_CLUBS = Card(
    name=CardName.KING, suit=Suit.CLUBS, value=4, value_atout=4, order=5, order_atout=3
)
ACE_CLUBS = Card(
    name=CardName.ACE, suit=Suit.CLUBS, value=11, value_atout=11, order=7, order_atout=5
)

cards: dict[Suit, dict[CardName, Card]] = {
    Suit.HEARTS: {
        CardName.ACE: ACE_HEARTS,
        CardName.KING: KING_HEARTS,
        CardName.QUEEN: QUEEN_HEARTS,
        CardName.JACK: JACK_HEARTS,
        CardName.TEN: TEN_HEARTS,
        CardName.NINE: NINE_HEARTS,
        CardName.EIGHT: EIGHT_HEARTS,
        CardName.SEVEN: SEVEN_HEARTS,
    },
    Suit.DIAMONDS: {
        CardName.ACE: ACE_DIAMONDS,
        CardName.KING: KING_DIAMONDS,
        CardName.QUEEN: QUEEN_DIAMONDS,
        CardName.JACK: JACK_DIAMONDS,
        CardName.TEN: TEN_DIAMONDS,
        CardName.NINE: NINE_DIAMONDS,
        CardName.EIGHT: EIGHT_DIAMONDS,
        CardName.SEVEN: SEVEN_DIAMONDS,
    },
    Suit.CLUBS: {
        CardName.ACE: ACE_CLUBS,
        CardName.KING: KING_CLUBS,
        CardName.QUEEN: QUEEN_CLUBS,
        CardName.JACK: JACK_CLUBS,
        CardName.TEN: TEN_CLUBS,
        CardName.NINE: NINE_CLUBS,
        CardName.EIGHT: EIGHT_CLUBS,
        CardName.SEVEN: SEVEN_CLUBS,
    },
    Suit.SPADES: {
        CardName.ACE: ACE_SPADES,
        CardName.KING: KING_SPADES,
        CardName.QUEEN: QUEEN_SPADES,
        CardName.JACK: JACK_SPADES,
        CardName.TEN: TEN_SPADES,
        CardName.NINE: NINE_SPADES,
        CardName.EIGHT: EIGHT_SPADES,
        CardName.SEVEN: SEVEN_SPADES,
    },
}


def create_deck() -> list[Card]:
    return [card for suit in Suit for card in cards[suit].values()]


def shuffle_deck(deck: list[Card]) -> list[Card]:
    shuffled = deck.copy()
    random.shuffle(shuffled)
    return shuffled


def deal_cards(deck: list[Card]) -> list[list[Card]]:
    """
    Deal cards to 4 players (8 each).

    This function distributes cards from the given deck to 4 players. Each player
    receives 8 cards in total, dealt in two rounds. In the first and second round, each player
    gets 3 cards, and in the third round, each player gets 2 cards.

    Args:
        deck (list[Card]): The deck of cards to be dealt.

    Returns:
        list[list[Card]]: A list containing 4 lists, each representing a player's hand.
    """
    hands: list[list[Card]] = [[] for _ in range(4)]
    for _ in range(2):
        for h in hands:
            for _ in range(3):
                h.append(deck.pop())
    for h in hands:
        for _ in range(2):
            h.append(deck.pop())
    return hands
