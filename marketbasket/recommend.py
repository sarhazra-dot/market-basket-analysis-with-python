"""Turn association rules into "customers also bought" recommendations."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .rules import format_itemset

RECOMMENDATION_COLUMNS = ["item", "score", "confidence", "lift", "support", "because"]


def known_items(rules: pd.DataFrame) -> set[str]:
    """All item names that appear in the rules."""
    items: set[str] = set()
    for column in ("antecedents", "consequents"):
        for itemset in rules[column]:
            items.update(itemset)
    return items


def recommend(
    rules: pd.DataFrame,
    basket: Iterable[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend items to add to ``basket``.

    A rule fires when all of its antecedent items are already in the basket.
    Only rules with a single-item consequent are used: the confidence and lift
    of a rule like ``A -> B + C`` describe buying B *and* C together, so they
    cannot be credited to B alone. Items not yet in the basket become
    candidates. Each candidate is scored with ``confidence * lift`` so that rules which are
    both reliable (confidence) and surprising (lift) rank first; when several
    rules suggest the same item the best-scoring one is kept.

    Item names are matched case-insensitively. Unknown items are ignored.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    vocabulary = {name.casefold(): name for name in known_items(rules)}
    basket_set = {vocabulary[b.strip().casefold()] for b in basket if b.strip().casefold() in vocabulary}

    best: dict[str, dict] = {}
    for row in rules.itertuples(index=False):
        antecedents, consequents = row.antecedents, row.consequents
        if len(consequents) != 1 or not antecedents <= basket_set:
            continue
        score = float(row.confidence) * float(row.lift)
        for item in consequents - basket_set:
            if item not in best or score > best[item]["score"]:
                best[item] = {
                    "item": item,
                    "score": score,
                    "confidence": float(row.confidence),
                    "lift": float(row.lift),
                    "support": float(row.support),
                    "because": format_itemset(antecedents),
                }

    if not best:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    result = pd.DataFrame(best.values(), columns=RECOMMENDATION_COLUMNS)
    result = result.sort_values(["score", "item"], ascending=[False, True], kind="mergesort")
    return result.head(top_n).reset_index(drop=True)
