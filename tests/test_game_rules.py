from game import CoincheGame
from models import Bid, Player, Suit, CardName
from game_rules import GameRules
from deck import cards


def init_game_with_players():
    game = CoincheGame()
    game.add_player(Player(id=0, name="a", team=0))
    game.add_player(Player(id=1, name="b", team=1))
    game.add_player(Player(id=2, name="c", team=0))
    game.add_player(Player(id=3, name="d", team=1))
    game.start_game()
    return game


def test_is_valid_bid():
    assert (
        GameRules.is_valid_bid(Bid(points=90, suit=Suit.HEARTS, player=1), None) is True
    )
    assert (
        GameRules.is_valid_bid(Bid(points=70, suit=Suit.HEARTS, player=1), None)
        is False
    )
    assert (
        GameRules.is_valid_bid(
            Bid(points=90, suit=Suit.HEARTS, player=1),
            Bid(points=100, suit=Suit.SPADES, player=2),
        )
        is False
    )
    assert (
        GameRules.is_valid_bid(
            Bid(points=120, suit=Suit.HEARTS, player=1),
            Bid(points=110, suit=Suit.SPADES, player=2),
        )
        is True
    )


def test_is_coinche_valid():
    assert (
        GameRules.is_coinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 0)
        is True
    )
    assert (
        GameRules.is_coinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 1)
        is False
    )
    assert (
        GameRules.is_coinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 3)
        is False
    )


def test_is_surcoinche_valid():
    assert (
        GameRules.is_surcoinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 0)
        is True
    )
    assert (
        GameRules.is_surcoinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 1)
        is False
    )
    assert (
        GameRules.is_surcoinche_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 2)
        is True
    )


def test_is_pass_valid():
    assert (
        GameRules.is_pass_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 0) is True
    )
    assert (
        GameRules.is_pass_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 1) is False
    )
    assert (
        GameRules.is_pass_valid(Bid(points=90, suit=Suit.HEARTS, player=1), 2) is True
    )


def test_get_best_card_in_trick():
    heart_ace = cards[Suit.HEARTS][CardName.ACE]
    heart_king = cards[Suit.HEARTS][CardName.KING]
    heart_queen = cards[Suit.HEARTS][CardName.QUEEN]
    heart_jack = cards[Suit.HEARTS][CardName.JACK]
    spade_jack = cards[Suit.SPADES][CardName.JACK]
    spade_nine = cards[Suit.SPADES][CardName.NINE]
    spade_king = cards[Suit.SPADES][CardName.KING]
    spade_queen = cards[Suit.SPADES][CardName.QUEEN]
    club_ace = cards[Suit.CLUBS][CardName.ACE]

    # 1 seule carte
    trick = [heart_ace]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 2 cartes avec les deux cartes de la même couleur différente de l'atout
    trick = [heart_ace, heart_king]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 2 cartes à l'atout
    trick = [spade_jack, spade_nine]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 2 cartes avec une de la couleur et une d'une autre couleur mais différente de l'atout
    trick = [heart_queen, club_ace]
    assert GameRules.get_best_card_in_trick(trick, Suit.DIAMONDS) == heart_queen

    # 2 cartes avec une de la couleur et une de l'atout
    trick = [heart_ace, spade_jack]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 3 cartes de la même couleur différentes de l'atout
    trick = [heart_ace, heart_king, heart_queen]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 3 cartes à l'atout
    trick = [spade_nine, spade_queen, spade_jack]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 3 cartes dont une d'une couleur différentes mais sans atout
    trick = [heart_ace, heart_king, club_ace]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 3 cartes dont une à l'atout
    trick = [heart_ace, heart_king, spade_jack]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 3 cartes dont une à l'atout et une d'une autre couleur
    trick = [heart_ace, spade_jack, club_ace]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 4 cartes à la même couleur différentes de l'atout
    trick = [heart_ace, heart_king, heart_queen, heart_jack]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 4 cartes à l'atout
    trick = [spade_nine, spade_queen, spade_jack, spade_king]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack

    # 4 cartes dont une d'une autre couleur
    trick = [heart_ace, heart_king, heart_queen, club_ace]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == heart_ace

    # 4 cartes dont une à l'atout
    trick = [heart_ace, heart_king, spade_jack, heart_queen]
    assert GameRules.get_best_card_in_trick(trick, Suit.SPADES) == spade_jack


def test_determine_trick_winner():
    game = init_game_with_players()
    heart_ace = cards[Suit.HEARTS][CardName.ACE]
    heart_king = cards[Suit.HEARTS][CardName.KING]
    heart_queen = cards[Suit.HEARTS][CardName.QUEEN]
    heart_jack = cards[Suit.HEARTS][CardName.JACK]
    spade_jack = cards[Suit.SPADES][CardName.JACK]
    spade_nine = cards[Suit.SPADES][CardName.NINE]
    spade_king = cards[Suit.SPADES][CardName.KING]
    spade_queen = cards[Suit.SPADES][CardName.QUEEN]
    club_ace = cards[Suit.CLUBS][CardName.ACE]

    trick = [heart_ace]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 2 cartes avec les deux cartes de la même couleur différente de l'atout
    trick = [heart_ace, heart_king]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 2 cartes à l'atout
    trick = [spade_jack, spade_nine]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 2 cartes avec une de la couleur et une d'une autre couleur mais différente de l'atout
    trick = [heart_queen, club_ace]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 2 cartes avec une de la couleur et une de l'atout
    trick = [heart_ace, spade_jack]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[1]
    )

    # 3 cartes de la même couleur différentes de l'atout
    trick = [heart_ace, heart_king, heart_queen]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 3 cartes à l'atout
    trick = [spade_nine, spade_queen, spade_jack]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[2]
    )

    # 3 cartes dont une d'une couleur différentes mais sans atout
    trick = [heart_ace, heart_king, club_ace]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 3 cartes dont une à l'atout
    trick = [heart_ace, heart_king, spade_jack]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[2]
    )

    # 3 cartes dont une à l'atout et une d'une autre couleur
    trick = [heart_ace, spade_jack, club_ace]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[1]
    )

    # 4 cartes à la même couleur différentes de l'atout
    trick = [heart_ace, heart_king, heart_queen, heart_jack]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 4 cartes à l'atout
    trick = [spade_nine, spade_queen, spade_jack, spade_king]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[2]
    )

    # 4 cartes dont une d'une autre couleur
    trick = [heart_ace, heart_king, heart_queen, club_ace]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[0]
    )

    # 4 cartes dont une à l'atout
    trick = [heart_ace, heart_king, spade_jack, heart_queen]
    assert (
        GameRules.determine_trick_winner(trick, Suit.SPADES, game.players)
        == game.players[2]
    )


def test_is_play_valid():
    """
    Test the is_play_valid method of the GameRules class.
    """
    game = init_game_with_players()
    heart_ace = cards[Suit.HEARTS][CardName.ACE]
    heart_king = cards[Suit.HEARTS][CardName.KING]
    heart_queen = cards[Suit.HEARTS][CardName.QUEEN]
    heart_jack = cards[Suit.HEARTS][CardName.JACK]
    spade_jack = cards[Suit.SPADES][CardName.JACK]
    spade_nine = cards[Suit.SPADES][CardName.NINE]
    spade_queen = cards[Suit.SPADES][CardName.QUEEN]
    club_ace = cards[Suit.CLUBS][CardName.ACE]
    diamond_ace = cards[Suit.DIAMONDS][CardName.ACE]
    diamond_king = cards[Suit.DIAMONDS][CardName.KING]

    # Player with no cards in trick
    hand = [heart_king, heart_queen, spade_jack, spade_nine, spade_queen, club_ace]

    player = game.players[0]
    player.hand = hand
    # Play in empty trick
    assert (
        GameRules.is_play_valid(heart_king, player, [], Suit.SPADES, game.players)
        is True
    )

    # Play same suit with one card in trick
    player = game.players[1]
    player.hand = hand
    trick = [heart_ace]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is True
    )

    # try to play a card not in player's hand
    assert (
        GameRules.is_play_valid(diamond_ace, player, trick, Suit.SPADES, game.players)
        is False
    )

    # try to play another suit with an atout in hand
    trick = [diamond_ace]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is False
    )

    # play an atout because no other card in the same suit
    assert (
        GameRules.is_play_valid(spade_jack, player, trick, Suit.SPADES, game.players)
        is True
    )

    # Player with two cards in trick, same suit
    player = game.players[2]
    player.hand = hand
    trick = [heart_ace, heart_jack]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is True
    )

    # Player with two cards in trick, different suits, no trump
    trick = [diamond_ace, club_ace]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is True
    )

    # Player with two cards in trick, different suits, has trump
    assert (
        GameRules.is_play_valid(spade_jack, player, trick, Suit.SPADES, game.players)
        is True
    )

    # try to play a lower atout card even with a higher atout card in hand
    trick = [diamond_ace, spade_nine]
    assert (
        GameRules.is_play_valid(spade_queen, player, trick, Suit.SPADES, game.players)
        is False
    )

    # Player with three cards in trick, same suit
    player = game.players[3]
    player.hand = hand
    trick = [heart_ace, heart_jack, heart_queen]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is True
    )

    # Player with three cards in trick, different suits, no trump
    trick = [diamond_ace, club_ace, diamond_king]
    assert (
        GameRules.is_play_valid(heart_king, player, trick, Suit.SPADES, game.players)
        is False
    )

    # Player with three cards in trick, different suits, has trump
    assert (
        GameRules.is_play_valid(spade_jack, player, trick, Suit.SPADES, game.players)
        is True
    )

    # try to play a lower atout card even with a higher atout card in hand
    trick = [diamond_ace, spade_nine, club_ace]
    assert (
        GameRules.is_play_valid(spade_queen, player, trick, Suit.SPADES, game.players)
        is False
    )

    # Player with three cards in trick, partner has best card
    trick = [diamond_ace, diamond_king]
    player = game.players[2]
    player.hand = [spade_queen, spade_nine, spade_jack, club_ace]
    assert (
        GameRules.is_play_valid(club_ace, player, trick, Suit.SPADES, game.players)
        is True
    )

    # Player with three cards in trick, partner does not have best card
    trick = [diamond_ace, diamond_king, spade_nine]
    player = game.players[3]
    assert (
        GameRules.is_play_valid(spade_jack, player, trick, Suit.SPADES, game.players)
        is True
    )
