import random
from itertools import product

from models import GameStage, Player, Bid, Suit
from game import CoincheGame
from game_rules import GameRules
from ai.checkpoint import CheckpointManager
from ai.metrics import EpisodeMetrics, MetricsTracker
from ai.models import CoincheAgent
from ai.training import CoincheTrainer
from ai.monitoring import NetworkMonitor
from ai.utils import Experience, calculate_reward


def main():
    # Initialize game and AI agents
    players = [Player(id=i, name=f"Player {i}", team=i % 2) for i in range(4)]
    game = CoincheGame()
    monitor = NetworkMonitor()

    for player in players:
        game.add_player(player)

    # Create AI agents for each player
    agents = {i: CoincheAgent() for i in range(len(game.players))}
    trainers = {i: CoincheTrainer(agent) for i, agent in agents.items()}
    metrics_tracker = MetricsTracker()
    checkpoint_manager = CheckpointManager()
    try:
        for agent in agents.values():
            checkpoint_manager.load_checkpoint(agent)
        metrics_tracker.load_metrics()
        print("Resumed from previous checkpoint")
    except FileNotFoundError:
        print("Starting new training session")
    # Training loop

    # Training loop
    num_episodes = 1000
    save_frequency = 50  # Save every 50 episodes
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        episode_rewards = {i: 0.0 for i in range(4)}
        contracts_won = 0
        contracts_lost = 0
        total_points = 0
        for mene in range(10):
            print(f"Episode {episode + 1} Game {mene + 1}")
            # Bidding phase
            while game.phase == GameStage.BID:
                for i1 in range(4):
                    current_player = game.current_player
                    player = game.get_current_player()
                    if current_player != i1:
                        continue

                    # Generate random to compare AI
                    if i1 == 2:
                        bid = choose_random_bid(i1, game)
                    else:
                        bid = agents[i1].select_bid(game, i1)
                    if bid.is_pass:
                        game.pass_bid(player)
                    else:
                        game.place_bid(bid)
                    if game.phase == GameStage.GAME:  # type: ignore
                        break
            game.end_bidding()
            if game.current_bid is None:
                continue

            for _ in range(8):  # 8 tricks
                old_game = CoincheGame(**game.__dict__)
                for i2 in range(4):  # 4 players per trick
                    current_player = game.current_player
                    player = game.get_current_player()
                    # Generate random to compare AI
                    card = agents[i2].select_card(game, current_player)
                    game.play_card(player, card)

                    # Store experience
                if len(game.tricks) == 0:
                    break
                for i3, _ in enumerate(game.players):
                    card = game.tricks[-1][i3]
                    reward = calculate_reward(game, i3)

                    trainers[i3].replay_buffer.push(
                        Experience(
                            game=old_game,
                            action=card,
                            reward=reward,
                            next_game=game,
                        )
                    )
                    episode_rewards[i3] += reward
            if len(game.tricks) == 0:
                continue
            attack_points = game.logs[-1].attack_points
            if not game.logs[-1].bid.points:
                raise ValueError("Bid not valid")
            total_points = (
                (-1 if attack_points < game.logs[-1].bid.points else 1)
                * game.logs[-1].bid.points
                * (2 if game.logs[-1].bid.is_coinche else 1)
            )
            if total_points >= 0:
                contracts_won += 1
            else:
                contracts_lost += 1

            # Update networks
            for trainer in trainers.values():
                trainer.update_networks()
                # input()
            # Store experience
            for agent in agents.values():
                stats = monitor.update(agent.state_encoder)
                stats.update(vars(agent.bidding_network))
                stats.update(vars(agent.card_play_network))

                # Check for anomalies or issues
                if stats.get("anomalies"):
                    print(f"Training anomalies detected: {stats['anomalies']}")

            metrics = EpisodeMetrics(
                episode=episode,
                episode_rewards=episode_rewards,
                total_reward=sum(episode_rewards.values()),
                win_rate=contracts_won / (contracts_won + contracts_lost)
                if contracts_won + contracts_lost > 0
                else 0,
                average_points=total_points / (contracts_won + contracts_lost)
                if contracts_won + contracts_lost > 0
                else 0,
                successful_contracts=contracts_won,
                failed_contracts=contracts_lost,
            )
            metrics_tracker.add_episode_metrics(metrics)

            game.new_game()

        # Save checkpoints and metrics periodically
        if (episode + 1) % save_frequency == 0:
            metrics_tracker.save_metrics()
            metrics_tracker.plot_metrics(
                f"src/ai/training_metrics/episode_{episode+1}_plot.png"
            )
            for i5, agent in agents.items():
                checkpoint_manager.save_checkpoint(
                    agent, episode, {"rewards": episode_rewards[i5]}
                )
            print(f"Saved checkpoint at episode {episode + 1}")


def choose_random_bid(player_id: int, game: CoincheGame) -> Bid:
    bids_possible = [
        Bid(player=id, points=points, suit=suit)
        for id, points, suit in product(
            [player_id],
            list(range(80, 120, 10)) + list(range(115, 165, 5)),
            list(Suit),
        )
        if GameRules.is_valid_bid(
            Bid(player=player_id, points=points, suit=suit), game.current_bid
        )
    ]
    return random.sample(
        bids_possible
        + [
            Bid(player=player_id, is_pass=True, points=None, suit=None)
            for _ in range(len(bids_possible) // 2 + 1)
        ],
        k=1,
    )[0]


if __name__ == "__main__":
    main()
