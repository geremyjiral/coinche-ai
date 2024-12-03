import random
from game import CoincheGame
from models import Bid, Player, Suit, GameStage
from game_rules import GameRules
from itertools import product


def simulate_bidding(game: CoincheGame):
    for _ in range(random.randint(4, 7)):
        player = game.get_current_player()
        bids_possible = [
            Bid(player=id, points=points, suit=suit)
            for id, points, suit in product(
                [player.id],
                list(range(80, 120, 10)) + list(range(115, 165, 5)),
                list(Suit),
            )
            if GameRules.is_valid_bid(
                Bid(player=player.id, points=points, suit=suit), game.current_bid
            )
        ]
        bid = random.sample(
            bids_possible
            + [
                Bid(player=player.id, is_pass=True, points=None, suit=None)
                for _ in range(len(bids_possible) + 1)
            ],
            k=1,
        )[0]
        if game.phase == GameStage.GAME:
            break
        if bid.is_pass:
            game.pass_bid(player)
        else:
            game.place_bid(bid)
    game.end_bidding()


def simulate_play(game: CoincheGame):
    for _ in range(8):
        for _ in range(4):
            player = game.get_current_player()
            card = random.choice(
                [
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
            )
            game.play_card(player, card)


def main():
    players = [Player(id=i, name=f"Player {i}", team=i % 2) for i in range(1, 5)]
    game = CoincheGame()

    for player in players:
        game.add_player(player)
    game.start_game()
    for _ in range(4):
        simulate_bidding(game)
        simulate_play(game)
        print(game.scores)
        print(game.logs[-1])
        game.new_game()


if __name__ == "__main__":
    main()
