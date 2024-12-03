from game import CoincheGame
from game_rules import GameRules


def calculate_reward(game: CoincheGame, player_id: int) -> float:
    """Compute reward for the current state from player's perspective"""
    if not game.atout or not game.current_bid or not game.current_bid.points:
        raise ValueError("Bid not set")

    reward = 0.0

    # Reward for winning tricks
    if GameRules.determine_trick_winner(game.tricks[-1], game.atout, game.players) in [
        game.players[p_id] for p_id in game.teams[game.players[player_id].team]
    ]:
        reward += (
            1
            if player_id
            == GameRules.determine_trick_winner(
                game.tricks[-1], game.atout, game.players
            )
            else 0.5
        ) * sum(
            card.value if card.suit != game.atout else card.value_atout
            for card in game.tricks[-1]
        ) + (0 if len(game.tricks) < 8 else 10)
    else:
        reward -= sum(
            card.value if card.suit != game.atout else card.value_atout
            for card in game.tricks[-1]
        ) + (0 if len(game.tricks) < 8 else 10)

    # Additional reward for completing game
    if len(game.tricks) == 8:
        points = sum(
            card.value if card.suit != game.atout else card.value_atout
            for trick in game.tricks
            for card in trick
            if GameRules.determine_trick_winner(trick, game.atout, game.players)
            in [
                game.players[player_id]
                for player_id in game.teams[game.players[player_id].team]
            ]
        )

        is_player_attack = (
            game.current_bid.player in game.teams[game.players[player_id].team]
        )
        is_win = points >= game.current_bid.points

        reward += 3 * (
            (
                1
                if (is_player_attack and is_win)
                or (not is_player_attack and not is_win)
                else -1
            )
            * game.current_bid.points
            * (2 if game.current_bid.is_coinche else 1)
        )

    return reward
