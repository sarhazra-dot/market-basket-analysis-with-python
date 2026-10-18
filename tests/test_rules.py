import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import association_rules

from marketbasket.preprocess import encode_baskets
from marketbasket.rules import (
    RULE_COLUMNS,
    format_itemset,
    generate_rules,
    load_rules,
    mine_frequent_itemsets,
    parse_itemset,
    prune_redundant_rules,
    save_itemsets,
    save_rules,
    top_rules,
)


def _toy_itemsets(toy_baskets, min_support=0.2):
    return mine_frequent_itemsets(encode_baskets(toy_baskets), min_support=min_support, max_len=None)


def _find(rules, antecedents, consequents):
    hit = rules[(rules["antecedents"] == frozenset(antecedents)) & (rules["consequents"] == frozenset(consequents))]
    assert len(hit) == 1, f"expected exactly one rule {antecedents} -> {consequents}"
    return hit.iloc[0]


# ------------------------------- itemset mining ------------------------------ #
def test_itemset_supports_match_hand_computation(toy_baskets):
    itemsets = _toy_itemsets(toy_baskets)
    support = dict(zip(itemsets["itemsets"], itemsets["support"]))
    assert support[frozenset({"bread"})] == pytest.approx(0.6)
    assert support[frozenset({"eggs"})] == pytest.approx(0.2)
    assert support[frozenset({"bread", "butter"})] == pytest.approx(0.4)
    assert support[frozenset({"bread", "butter", "milk"})] == pytest.approx(0.2)
    assert (itemsets["length"] == itemsets["itemsets"].apply(len)).all()
    assert itemsets["support"].is_monotonic_decreasing


def test_min_support_excludes_rare_itemsets(toy_baskets):
    itemsets = _toy_itemsets(toy_baskets, min_support=0.5)
    assert set(itemsets["itemsets"]) == {frozenset({"bread"}), frozenset({"butter"}), frozenset({"milk"})}


def test_max_len_limits_itemset_size(toy_baskets):
    itemsets = mine_frequent_itemsets(encode_baskets(toy_baskets), min_support=0.2, max_len=2)
    assert itemsets["length"].max() == 2


def test_apriori_and_fpgrowth_agree(synthetic_onehot):
    a = mine_frequent_itemsets(synthetic_onehot, 0.02, algorithm="apriori", max_len=3)
    f = mine_frequent_itemsets(synthetic_onehot, 0.02, algorithm="fpgrowth", max_len=3)
    sa = dict(zip(a["itemsets"], a["support"]))
    sf = dict(zip(f["itemsets"], f["support"]))
    assert sa.keys() == sf.keys()
    for key in sa:
        assert sa[key] == pytest.approx(sf[key])


def test_mining_validation(toy_baskets):
    onehot = encode_baskets(toy_baskets)
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            mine_frequent_itemsets(onehot, min_support=bad)
    with pytest.raises(ValueError):
        mine_frequent_itemsets(onehot, algorithm="nope")
    with pytest.raises(ValueError):
        mine_frequent_itemsets(onehot, max_len=0)


def test_mining_empty_inputs(toy_baskets):
    empty = encode_baskets(toy_baskets, min_item_support=0.99)  # zero columns
    out = mine_frequent_itemsets(empty, 0.1)
    assert out.empty and list(out.columns) == ["support", "itemsets", "length"]
    assert generate_rules(out).empty


# ---------------------------------- rule metrics ----------------------------- #
def test_rule_metrics_match_hand_computation(toy_baskets):
    rules = generate_rules(_toy_itemsets(toy_baskets), min_confidence=0.0, min_lift=0.0)
    assert len(rules) == 14  # 4 pairs * 2 + 1 triple * 6

    r = _find(rules, {"bread"}, {"butter"})
    assert r["support"] == pytest.approx(0.4)
    assert r["antecedent_support"] == pytest.approx(0.6)
    assert r["consequent_support"] == pytest.approx(0.6)
    assert r["confidence"] == pytest.approx(2 / 3)
    assert r["lift"] == pytest.approx((2 / 3) / 0.6)
    assert r["leverage"] == pytest.approx(0.4 - 0.36)
    assert r["conviction"] == pytest.approx(0.4 / (1 / 3))

    r = _find(rules, {"eggs"}, {"milk"})
    assert r["confidence"] == pytest.approx(1.0)
    assert r["lift"] == pytest.approx(1 / 0.6)
    assert math.isinf(r["conviction"])

    r = _find(rules, {"butter", "milk"}, {"bread"})
    assert r["confidence"] == pytest.approx(1.0)
    assert r["lift"] == pytest.approx(1 / 0.6)


def test_default_thresholds_filter_rules(toy_baskets):
    rules = generate_rules(_toy_itemsets(toy_baskets))  # confidence >= 0.3, lift >= 1
    found = {(frozenset(a), frozenset(c)) for a, c in zip(rules["antecedents"], rules["consequents"])}
    # Worked out by hand from the supports in the toy data (n = 5):
    expected = {
        ("bread", "butter"), ("butter", "bread"),   # conf 2/3, lift 1.11
        ("bread", "milk"), ("milk", "bread"),       # conf 2/3, lift 1.11
        ("eggs", "milk"), ("milk", "eggs"),         # lift 1.67
    }
    expected = {(frozenset({a}), frozenset({c})) for a, c in expected}
    expected |= {
        (frozenset({"butter", "milk"}), frozenset({"bread"})),   # conf 1.0, lift 1.67
        (frozenset({"bread"}), frozenset({"butter", "milk"})),   # conf 1/3, lift 1.67
    }
    assert found == expected
    assert (rules["confidence"] >= 0.3 - 1e-9).all()
    assert (rules["lift"] >= 1.0 - 1e-9).all()
    # butter -> milk has lift 0.56 and butter -> bread + milk has lift 0.83: both excluded
    assert (frozenset({"butter"}), frozenset({"milk"})) not in found
    assert (frozenset({"butter"}), frozenset({"bread", "milk"})) not in found


def test_antecedent_and_consequent_are_disjoint_and_nonempty(synthetic_onehot):
    itemsets = mine_frequent_itemsets(synthetic_onehot, 0.02, max_len=3)
    rules = generate_rules(itemsets, min_confidence=0.0, min_lift=0.0)
    assert len(rules) > 0
    for a, c in zip(rules["antecedents"], rules["consequents"]):
        assert a and c and not (a & c)


def test_rules_sorted_by_lift_then_confidence(synthetic_onehot):
    rules = generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.02, max_len=3))
    assert list(rules.columns) == RULE_COLUMNS
    assert rules["lift"].is_monotonic_decreasing


def test_results_are_deterministic(synthetic_onehot):
    def run():
        return generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.02, max_len=3))

    first, second = run(), run()
    assert [format_itemset(a) for a in first["antecedents"]] == [format_itemset(a) for a in second["antecedents"]]
    assert first["lift"].tolist() == second["lift"].tolist()


def test_metrics_agree_with_mlxtend(synthetic_onehot):
    """Cross-check support/confidence/lift/leverage against mlxtend itself."""
    itemsets = mine_frequent_itemsets(synthetic_onehot, 0.02, max_len=3)
    mine = generate_rules(itemsets, min_confidence=0.0, min_lift=0.0)

    kwargs = {"metric": "confidence", "min_threshold": 0.0}
    if "num_itemsets" in inspect.signature(association_rules).parameters:
        kwargs["num_itemsets"] = len(synthetic_onehot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = association_rules(itemsets[["support", "itemsets"]], **kwargs)

    assert len(mine) == len(reference)
    ref = {(r.antecedents, r.consequents): r for r in reference.itertuples()}
    for r in mine.itertuples():
        other = ref[(r.antecedents, r.consequents)]
        assert r.support == pytest.approx(other.support)
        assert r.confidence == pytest.approx(other.confidence)
        assert r.lift == pytest.approx(other.lift)
        assert r.leverage == pytest.approx(other.leverage, abs=1e-12)


def test_rule_validation(toy_baskets):
    itemsets = _toy_itemsets(toy_baskets)
    with pytest.raises(ValueError):
        generate_rules(itemsets, min_confidence=1.2)
    with pytest.raises(ValueError):
        generate_rules(itemsets, min_lift=-1)


# ------------------------- planted associations are found --------------------- #
def test_planted_associations_are_recovered(synthetic_onehot):
    rules = generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.01, max_len=3), min_confidence=0.0, min_lift=0.0)
    pasta = _find(rules, {"Spaghetti"}, {"Tomato Sauce"})
    assert pasta["confidence"] > 0.7 and pasta["lift"] > 3
    baby = _find(rules, {"Diapers"}, {"Baby Wipes"})
    assert baby["confidence"] > 0.6 and baby["lift"] > 3
    movie = _find(rules, {"Potato Chips"}, {"Cola"})
    assert movie["lift"] > 3


def test_popular_but_independent_item_has_lift_near_one(synthetic_onehot):
    """Milk is bought often, but not *because of* Spaghetti: lift ~ 1."""
    rules = generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.01, max_len=2), min_confidence=0.0, min_lift=0.0)
    r = _find(rules, {"Spaghetti"}, {"Milk"})
    assert 0.75 < r["lift"] < 1.25


# ------------------------------- display helpers ------------------------------ #
def _manual_rules(rows):
    return pd.DataFrame(
        [
            {"antecedents": frozenset(a), "consequents": frozenset(c), "confidence": conf, "lift": lift}
            for a, c, conf, lift in rows
        ]
    )


def test_prune_redundant_rules():
    rules = _manual_rules(
        [
            ({"A"}, {"C"}, 0.50, 2.0),
            ({"A", "B"}, {"C"}, 0.52, 2.1),  # +0.02: redundant
            ({"A", "D"}, {"C"}, 0.60, 2.4),  # +0.10: informative
            ({"X"}, {"C"}, 0.30, 1.2),       # different antecedent: unrelated
            ({"A"}, {"Z"}, 0.40, 1.5),       # different consequent: unrelated
        ]
    )
    kept = prune_redundant_rules(rules, min_confidence_gain=0.05)
    names = [(format_itemset(a), format_itemset(c)) for a, c in zip(kept["antecedents"], kept["consequents"])]
    assert ("A | B", "C") not in names
    assert ("A | D", "C") in names
    assert len(kept) == 4
    # gain 0 keeps ties and improvements but drops strictly worse specialisations
    worse = _manual_rules([({"A"}, {"C"}, 0.5, 2.0), ({"A", "B"}, {"C"}, 0.4, 1.6)])
    assert len(prune_redundant_rules(worse, 0.0)) == 1
    with pytest.raises(ValueError):
        prune_redundant_rules(rules, -0.1)
    assert prune_redundant_rules(rules.iloc[0:0]).empty


def test_top_rules_one_per_itemset_and_limit():
    rules = _manual_rules(
        [
            ({"A"}, {"B"}, 0.9, 5.0),
            ({"B"}, {"A"}, 0.5, 5.0),  # same itemset as above, worse confidence
            ({"C"}, {"D"}, 0.8, 4.0),
            ({"E"}, {"F"}, 0.7, 3.0),
        ]
    )
    top = top_rules(rules, n=2)
    assert len(top) == 2
    assert top.iloc[0]["antecedents"] == frozenset({"A"})
    assert top.iloc[1]["antecedents"] == frozenset({"C"})
    assert len(top_rules(rules, n=10, one_per_itemset=False)) == 4
    assert top_rules(rules.iloc[0:0]).empty


# ---------------------------------- persistence ------------------------------ #
def test_itemset_text_roundtrip_with_commas():
    items = frozenset({"Salt, fine", "Pepper"})
    assert parse_itemset(format_itemset(items)) == items


def test_rules_csv_roundtrip(tmp_path, toy_baskets):
    rules = generate_rules(_toy_itemsets(toy_baskets), min_confidence=0.0, min_lift=0.0)
    path = save_rules(rules, tmp_path / "nested" / "rules.csv")
    loaded = load_rules(path)
    assert len(loaded) == len(rules)
    assert loaded["antecedents"].tolist() == rules["antecedents"].tolist()
    assert loaded["consequents"].tolist() == rules["consequents"].tolist()
    np.testing.assert_allclose(loaded["lift"], rules["lift"])
    assert np.isinf(loaded["conviction"]).sum() == np.isinf(rules["conviction"]).sum() > 0


def test_load_rules_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rules(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_rules(bad)


def test_save_itemsets_writes_text(tmp_path, toy_baskets):
    path = save_itemsets(_toy_itemsets(toy_baskets), tmp_path / "i.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == ["support", "itemsets", "length"]
    assert "bread | butter" in set(df["itemsets"])
