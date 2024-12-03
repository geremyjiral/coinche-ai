from deck import create_deck, shuffle_deck, deal_cards


def test_deal_cards():
    deck = create_deck()
    shuffled_deck = shuffle_deck(deck)
    hands = deal_cards(shuffled_deck)

    # Check that there are 4 hands
    assert len(hands) == 4

    # Check that each hand has 8 cards
    for hand in hands:
        assert len(hand) == 8

    # Check that all cards are unique across all hands
    all_cards = [card for hand in hands for card in hand]
    assert len(all_cards) == len(set(all_cards))

    # Check that the total number of cards dealt is 32
    assert len(all_cards) == 32

    # Check that the deck is now empty
    assert len(shuffled_deck) == 0
