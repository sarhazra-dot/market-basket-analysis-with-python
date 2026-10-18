"""Turn cleaned transaction rows into baskets and a one-hot matrix."""

from __future__ import annotations

from collections import Counter

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder


def build_baskets(df: pd.DataFrame) -> list[list[str]]:
    """Group rows into baskets: one sorted list of unique items per invoice.

    Baskets are ordered by invoice id so results are deterministic.
    """
    if not {"invoice", "item"} <= set(df.columns):
        raise ValueError("DataFrame must have 'invoice' and 'item' columns")

    grouped: dict[str, set[str]] = {}
    for invoice, item in zip(df["invoice"], df["item"]):
        grouped.setdefault(invoice, set()).add(item)
    return [sorted(items) for _, items in sorted(grouped.items())]


def encode_baskets(
    baskets: list[list[str]],
    min_item_support: float = 0.0,
    max_items: int | None = None,
) -> pd.DataFrame:
    """One-hot encode baskets into a boolean DataFrame (rows = baskets).

    ``min_item_support`` drops items that appear in too few baskets to ever be
    part of a frequent itemset. This is a lossless speed/memory optimisation
    when it equals the mining ``min_support``: supports are still computed
    against the *full* number of baskets, because no basket is removed (a
    basket that loses all its items stays as an all-False row).

    ``max_items`` optionally keeps only the N most frequent items. Unlike the
    support filter this can change results, so it is off by default.
    """
    n_baskets = len(baskets)
    if n_baskets == 0:
        raise ValueError("No baskets to encode")
    if not 0.0 <= min_item_support <= 1.0:
        raise ValueError("min_item_support must be between 0 and 1")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be >= 1")

    counts = Counter(item for basket in baskets for item in set(basket))
    # Small tolerance keeps the filter looser than the miner's, never stricter.
    keep = {item for item, c in counts.items() if c / n_baskets >= min_item_support - 1e-12}

    if max_items is not None:
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        keep &= {item for item, _ in ranked[:max_items]}

    filtered = [[item for item in basket if item in keep] for basket in baskets]
    encoder = TransactionEncoder()
    matrix = encoder.fit(filtered).transform(filtered)
    return pd.DataFrame(matrix, columns=encoder.columns_)
