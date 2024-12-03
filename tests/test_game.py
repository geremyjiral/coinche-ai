import pytest
from game import CoincheGame
from game_rules import GameRules
from models import Card, Bid, CardName, Suit, Player, GameStage
from deck import cards
import random


def init_game_with_players():
    game = CoincheGame()
    game.add_player(Player(id=0, name="a", team=0))
    game.add_player(Player(id=1, name="b", team=1))
    game.add_player(Player(id=2, name="c", team=0))
    game.add_player(Player(id=3, name="d", team=1))
    return game


def generate_random_trick(deck: list[Card]) -> list[Card]:
    trick: list[Card] = []
    for c in random.sample(deck, k=4):
        trick.append(c)
    return trick


def generate_random_game_tricks(deck: list[Card]) -> list[list[Card]]:
    tricks: list[list[Card]] = []
    for _ in range(8):
        trick = generate_random_trick(deck)
        tricks.append(trick)
        deck = [c for c in deck if c not in trick]
    return tricks


def generate_capot(deck: list[Card], atout: Suit) -> list[list[Card]]:
    tricks: list[list[Card]] = [[c] for c in cards[atout].values()]
    for c in cards[atout].values():
        deck.remove(c)
    for i in range(8):
        for c in random.sample(deck, k=3):
            tricks[i].append(c)
            deck.remove(c)
    return tricks


def test_calculate_scores_no_bid():
    game = CoincheGame()
    with pytest.raises(ValueError, match="No bid placed"):
        game.calculate_scores()


def test_calculate_scores_bid_team_wins():
    game = init_game_with_players()

    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    game.tricks = generate_capot(game.deck.copy(), game.current_bid.suit)  # type: ignore

    game.calculate_scores()
    assert game.logs[-1].attack_points == 162
    assert game.logs[-1].defense_points == 0

    assert game.scores == [80, -80]


def test_calculate_scores_defend_team_wins():
    game = init_game_with_players()

    game.place_bid(Bid(player=1, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    game.tricks = generate_capot(game.deck.copy(), game.current_bid.suit)  # type: ignore

    game.calculate_scores()
    assert game.logs[-1].attack_points == 0
    assert game.logs[-1].defense_points == 162

    assert game.scores == [80, -80]


def test_calculate_scores_coinche():
    game = init_game_with_players()

    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS, is_coinche=True))
    game.end_bidding()

    game.tricks = generate_capot(game.deck.copy(), game.current_bid.suit)  # type: ignore

    game.calculate_scores()
    assert game.logs[-1].attack_points == 162
    assert game.logs[-1].defense_points == 0

    assert game.scores == [160, -160]


def test_add_player_to_team():
    game = CoincheGame()

    player1 = Player(id=0, team=0, name="Player 1")
    player2 = Player(id=1, team=1, name="Player 2")
    player3 = Player(id=2, team=0, name="Player 3")
    player4 = Player(id=3, team=1, name="Player 4")

    game.add_player(player1)
    game.add_player(player2)
    game.add_player(player3)
    game.add_player(player4)

    assert len(game.players) == 4
    assert game.players == [player1, player2, player3, player4]
    assert game.teams == [[0, 2], [1, 3]]


def test_add_player_to_full_team():
    game = CoincheGame()
    game.teams = [[0, 1], []]
    game.players = [
        Player(id=0, name="Player 1", team=0),
        Player(id=1, name="Player 2", team=0),
    ]

    player3 = Player(id=2, name="Player 3", team=0)

    with pytest.raises(ValueError, match="Team already has 2 players"):
        game.add_player(player3)


def test_add_player_starts_game():
    game = CoincheGame()
    game.teams = [[], []]
    game.players = []

    player1 = Player(id=0, name="Player 1", team=0)
    player2 = Player(id=1, name="Player 2", team=1)
    player3 = Player(id=2, name="Player 3", team=0)
    player4 = Player(id=3, name="Player 4", team=1)

    game.add_player(player1)
    game.add_player(player2)
    game.add_player(player3)
    game.add_player(player4)

    assert game.phase == GameStage.BID


def test_place_bid_valid():
    game = init_game_with_players()

    bid = Bid(player=0, points=80, suit=Suit.HEARTS)

    game.place_bid(bid)

    assert game.current_bid == bid
    assert game.bids[-1] == bid
    assert game.current_player == 1


def test_place_bid_invalid():
    game = init_game_with_players()
    bid = Bid(player=0, points=None, suit=Suit.HEARTS)

    with pytest.raises(ValueError, match="Invalid bid"):
        game.place_bid(bid)

    assert game.current_bid is None
    assert len(game.bids) == 0
    assert game.current_player == 0


def test_pass_bid_valid():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))

    player = game.players[game.current_player]
    game.pass_bid(player)

    assert len(game.bids) == 2
    assert game.bids[-1].is_pass
    assert game.current_player == 2


def test_pass_bid_invalid():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))

    player = game.players[0]
    with pytest.raises(ValueError, match="Invalid pass"):
        game.pass_bid(player)

    assert len(game.bids) == 1
    assert game.current_player == 1


def test_pass_bid_ends_bidding():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    for _ in range(3):
        player = game.players[game.current_player]
        game.pass_bid(player)

    assert len(game.bids) == 4
    assert game.bids[-1].is_pass
    assert game.phase == GameStage.GAME


def test_coinche_valid():
    game = init_game_with_players()

    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    player = game.players[1]
    game.coinche(player)

    assert game.current_bid is not None and game.current_bid.is_coinche
    assert game.phase == GameStage.GAME


def test_coinche_no_bid():
    game = init_game_with_players()

    player = game.players[1]
    with pytest.raises(ValueError, match="No bid placed, you can't coinche"):
        game.coinche(player)


def test_coinche_invalid():
    game = init_game_with_players()

    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    player = game.players[2]  # Invalid player for coinche
    with pytest.raises(ValueError, match="Invalid coinche"):
        game.coinche(player)


def test_end_bidding_no_bid():
    game = CoincheGame()
    with pytest.raises(ValueError, match="No bid placed"):
        game.end_bidding()


def test_end_bidding_sets_atout():
    game = init_game_with_players()

    bid = Bid(player=0, points=80, suit=Suit.HEARTS)
    game.place_bid(bid)
    game.end_bidding()

    assert game.atout == Suit.HEARTS


def test_end_bidding_sets_current_player():
    game = init_game_with_players()

    bid = Bid(player=0, points=80, suit=Suit.HEARTS)
    game.place_bid(bid)
    game.end_bidding()

    assert game.current_player == game.players[0].id


def test_end_bidding_sets_phase_to_game():
    game = init_game_with_players()

    bid = Bid(player=0, points=80, suit=Suit.HEARTS)
    game.place_bid(bid)
    game.end_bidding()

    assert game.phase == GameStage.GAME


def test_play_card_not_in_game_phase():
    game = init_game_with_players()

    player = game.players[0]
    card = cards[Suit.HEARTS][CardName.TEN]

    with pytest.raises(ValueError, match="Not in game phase"):
        game.play_card(player, card)


def test_play_card_not_players_turn():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()
    game.phase = GameStage.GAME

    player = game.players[1]

    card = cards[Suit.HEARTS][CardName.TEN]

    with pytest.raises(ValueError, match="Not this player's turn"):
        game.play_card(player, card)


def test_play_card_bid_not_valid():
    game = init_game_with_players()
    game.phase = GameStage.GAME

    player = game.players[0]

    card = cards[Suit.HEARTS][CardName.TEN]

    with pytest.raises(ValueError, match="Bid not valid"):
        game.play_card(player, card)


def test_play_card_invalid_play():
    game = init_game_with_players()
    game.start_game()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    player = game.players[game.current_player]
    card = next((c for c in player.hand))
    game.play_card(player, card)

    player = game.players[game.current_player]
    card = next(
        (c for suit in Suit for c in cards[suit].values() if c not in player.hand)
    )

    with pytest.raises(ValueError, match="Invalid play"):
        game.play_card(player, card)


def test_play_card_valid_play():
    game = init_game_with_players()
    game.start_game()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()
    game.phase = GameStage.GAME

    player = game.players[game.current_player]
    card = next((c for c in player.hand))
    game.play_card(player, card)

    assert card not in player.hand
    assert game.current_trick[-1] == card
    assert game.current_player == 1


def test_end_trick_no_bid():
    game = init_game_with_players()

    with pytest.raises(ValueError, match="No bid placed"):
        game.end_trick()


def test_end_trick_no_suit():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=None))

    with pytest.raises(ValueError, match="Bid not valid"):
        game.end_trick()


def test_end_trick_winner():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    game.current_trick = generate_random_trick(game.deck.copy())
    if game.current_bid is None or game.current_bid.suit is None:
        raise ValueError("Bid not valid")
    winning_player = GameRules.determine_trick_winner(
        game.current_trick, game.current_bid.suit, game.players
    )
    game.end_trick()

    assert game.current_trick == []
    assert game.players[game.current_player] == winning_player
    assert len(game.tricks) == 1


def test_end_trick_eight_tricks():
    game = init_game_with_players()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()

    game.tricks = generate_random_game_tricks(game.deck.copy())[:7]
    game.current_trick = generate_random_trick(game.deck.copy())
    game.end_trick()

    assert len(game.tricks) == 8
    assert game.phase == GameStage.BID


def test_get_current_player_initial():
    game = init_game_with_players()

    assert game.get_current_player().id == 0


def test_get_current_player_after_bidding():
    game = init_game_with_players()

    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.pass_bid(game.players[1])
    game.pass_bid(game.players[2])
    game.pass_bid(game.players[3])

    assert game.get_current_player().id == 0


def test_get_current_player_after_playing_card():
    game = init_game_with_players()
    game.start_game()
    game.place_bid(Bid(player=0, points=80, suit=Suit.HEARTS))
    game.end_bidding()
    game.phase = GameStage.GAME

    player = game.players[game.current_player]
    card = next((c for c in player.hand))
    game.play_card(player, card)

    assert game.get_current_player().id == 1


def test_new_game_rotates_players():
    game = init_game_with_players()
    initial_order = [player.id for player in game.players]

    game.new_game()
    new_order = [player.id for player in game.players]

    assert new_order == initial_order[1:] + initial_order[:1]


def test_new_game_starts_game():
    game = init_game_with_players()
    game.new_game()

    assert game.phase == GameStage.BID
    assert len(game.players[0].hand) > 0  # Ensure cards are dealt
