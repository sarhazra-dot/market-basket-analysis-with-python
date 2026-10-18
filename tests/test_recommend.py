import pytest

from marketbasket.preprocess import encode_baskets
from marketbasket.recommend import RECOMMENDATION_COLUMNS, known_items, recommend
from marketbasket.rules import generate_rules, mine_frequent_itemsets


@pytest.fixture()
def toy_rules(toy_baskets):
    itemsets = mine_frequent_itemsets(encode_baskets(toy_baskets), min_support=0.2, max_len=None)
    return generate_rules(itemsets)


def test_recommends_items_not_in_basket(toy_rules):
    result = recommend(toy_rules, ["bread"])
    assert list(result.columns) == RECOMMENDATION_COLUMNS
    assert set(result["item"]) == {"butter", "milk"}
    assert "bread" not in set(result["item"])


def test_multi_item_consequent_rules_are_not_credited_to_one_item(toy_rules):
    # bread -> butter + milk exists with confidence 1/3, but bread -> butter has 2/3.
    row = recommend(toy_rules, ["bread"]).set_index("item").loc["butter"]
    assert row["confidence"] == pytest.approx(2 / 3)
    assert row["because"] == "bread"


def test_multi_item_antecedent_fires_only_when_all_items_present(toy_rules):
    both = recommend(toy_rules, ["butter", "milk"])
    assert both.iloc[0]["item"] == "bread"
    assert both.iloc[0]["because"] == "butter | milk"
    # With butter alone, the "butter + milk -> bread" rule must not fire.
    alone = recommend(toy_rules, ["butter"])
    assert "butter | milk" not in set(alone["because"])


def test_matching_is_case_insensitive_and_trims(toy_rules):
    result = recommend(toy_rules, ["  BREAD "])
    assert set(result["item"]) == {"butter", "milk"}


def test_unknown_basket_gives_empty_result(toy_rules):
    result = recommend(toy_rules, ["unicorn"])
    assert result.empty and list(result.columns) == RECOMMENDATION_COLUMNS
    assert recommend(toy_rules, []).empty


def test_top_n_limits_and_orders_by_score(toy_rules):
    result = recommend(toy_rules, ["bread"], top_n=1)
    assert len(result) == 1
    full = recommend(toy_rules, ["bread"], top_n=5)
    assert full["score"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        recommend(toy_rules, ["bread"], top_n=0)


def test_known_items(toy_rules):
    assert known_items(toy_rules) == {"bread", "butter", "milk", "eggs"}


def test_recommendations_on_synthetic_data(synthetic_onehot):
    rules = generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.01, max_len=3))
    result = recommend(rules, ["Spaghetti"], top_n=3)
    assert result.iloc[0]["item"] == "Tomato Sauce"
    result = recommend(rules, ["Diapers"], top_n=3)
    assert result.iloc[0]["item"] == "Baby Wipes"
