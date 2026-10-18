import pandas as pd
import pytest

from marketbasket.rules import generate_rules, mine_frequent_itemsets
from marketbasket.viz import (
    plot_lift_heatmap,
    plot_support_confidence,
    plot_top_items,
    plot_top_rules,
)


@pytest.fixture(scope="module")
def rules(synthetic_onehot):
    return generate_rules(mine_frequent_itemsets(synthetic_onehot, 0.01, max_len=3))


def _is_png(path):
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size > 2000


def test_all_plots_are_written(tmp_path, synthetic_onehot, rules):
    assert _is_png(plot_top_items(synthetic_onehot, tmp_path / "a" / "items.png"))
    assert _is_png(plot_support_confidence(rules, tmp_path / "sc.png"))
    assert _is_png(plot_top_rules(rules, tmp_path / "top.png", top_n=8))
    heatmap = plot_lift_heatmap(rules, tmp_path / "heat.png", top_n=8)
    assert heatmap is not None and _is_png(heatmap)


def test_heatmap_returns_none_without_item_pairs(tmp_path, rules):
    multi_only = rules[rules["antecedents"].apply(len) > 1]
    assert plot_lift_heatmap(multi_only, tmp_path / "x.png") is None
    assert plot_lift_heatmap(rules.iloc[0:0], tmp_path / "y.png") is None
    assert not (tmp_path / "x.png").exists()


def test_top_items_with_few_items(tmp_path):
    onehot = pd.DataFrame({"a": [True, False], "b": [True, True]})
    assert _is_png(plot_top_items(onehot, tmp_path / "small.png", top_n=15))
