from models import Bid, Card, Player, Suit
from logger import get_logger

logger = get_logger(__name__)


class GameRules:
    @staticmethod
    def is_valid_bid(bid: Bid, current_bid: Bid | None) -> bool:
        if bid.points is None and not bid.is_pass:
            return False
        if not bid.points:
            return False
        if bid.points < 80:
            return False
        if current_bid and current_bid.points and bid.points <= current_bid.points:
            return False
        if bid.points <= 110 and bid.points % 10 != 0:
            return False
        if current_bid and bid.points > 110 and bid.points % 5 != 0:
            return False
        return True

    @staticmethod
    def is_coinche_valid(
        bid: Bid, player_id: int, player: Player, teams: list[list[int]]
    ) -> bool:
        return player_id not in teams[player.team]

    @staticmethod
    def is_surcoinche_valid(bid: Bid, player_id: int) -> bool:
        return player_id != bid.player

    @staticmethod
    def is_pass_valid(bid: Bid | None, player_id: int) -> bool:
        if not bid:
            logger.debug("no bid")
            return True

        logger.debug(
            f"bid player is not the player {player_id != bid.player} {player_id=} {bid.player=}"
        )
        return player_id != bid.player

    @staticmethod
    def get_best_card_in_trick(trick: list[Card], atout: Suit) -> Card:
        best_card = trick[0]
        trick_suit = trick[0].suit
        for card in trick[1:]:
            if card.suit == trick_suit:
                if trick_suit != atout:
                    if card.order > best_card.order and best_card.suit != atout:
                        best_card = card
                else:
                    if (
                        card.order_atout > best_card.order_atout
                        and best_card.suit == atout
                    ):
                        best_card = card
            else:
                if card.suit == atout:
                    if best_card.suit == atout:
                        if card.order_atout > best_card.order_atout:
                            best_card = card
                    else:
                        best_card = card
        return best_card

    @classmethod
    def determine_trick_winner(
        cls, trick: list[Card], atout: Suit, players: list[Player]
    ) -> Player:
        best_card = cls.get_best_card_in_trick(trick, atout)
        return players[trick.index(best_card)]

    @classmethod
    def is_play_valid(
        cls,
        card: Card,
        player: Player,
        trick: list[Card],
        atout: Suit,
        players: list[Player],
    ) -> bool:
        # logger.setLevel("DEBUG")
        logger.debug(f"{card}, {player}, {trick}, {atout}")
        if not trick:
            logger.debug("trick is empty")
            return True
        if card not in player.hand:
            logger.debug("card not in player hand")
            return False
        trick_suit = trick[0].suit
        # Card and trick suit are the same
        if card.suit == trick_suit:
            logger.debug("card and trick suit are the same")
            if card.suit != atout:
                logger.debug("card suit is not atout")
                return True
            else:
                logger.debug("card suit is atout")
                logger.debug(
                    f"il existe une carte meilleure à l'atout dans la main {any(
                        c.order_atout
                        > cls.get_best_card_in_trick(trick, atout).order_atout
                        for c in player.hand
                        if c != card and c.suit == atout
                    )}"
                )
                return not (
                    # there exists a stronger trump card in the player's hand
                    any(
                        c.order_atout
                        > cls.get_best_card_in_trick(trick, atout).order_atout
                        for c in player.hand
                        if c != card and c.suit == atout
                    )
                    # and the played card is weaker than the best trump card
                    and card.order_atout
                    < cls.get_best_card_in_trick(trick, atout).order_atout
                )
        else:
            logger.debug("card and trick suit are different")
            # Player has a card of the trick suit
            if any(c.suit == trick_suit for c in player.hand):
                logger.debug("player has a card of the trick suit")
                return False
            # Player has a trump card
            if any(c.suit == atout for c in player.hand):
                logger.debug("player has a trump card")
                # if not atout
                if card.suit != atout:
                    logger.debug("card is not atout")
                    logger.debug("player partner has played the best card")
                    # if the player partner has played the best card, all cards are valid else not valid because the player must play atout
                    return (
                        cls.determine_trick_winner(trick, atout, players).team
                        == player.team
                    )

                else:
                    logger.debug("card is atout")
                    if (
                        any(
                            c.order_atout > card.order_atout
                            for c in player.hand
                            if c != card and c.suit == atout
                        )
                        and card.order_atout
                        < cls.get_best_card_in_trick(trick, atout).order_atout
                    ):
                        logger.debug(
                            "there is a stronger atout card in the player's hand"
                        )
                        return False
                    else:
                        logger.debug(
                            "there is no stronger atout card in the player's hand"
                        )
                        return True
            # if the player has no card of the trick suit and no trump card, all cards are valid
            logger.debug("player has no card of the trick suit and no trump card")
            return True
