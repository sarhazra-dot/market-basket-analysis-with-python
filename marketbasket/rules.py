"""Frequent itemset mining and association rule generation.

Itemsets are mined with mlxtend (FP-Growth or Apriori). Rule metrics are
computed here directly from itemset supports so the results do not depend on
the changing signature of ``mlxtend.frequent_patterns.association_rules``.

Metric definitions for a rule A -> C (supports are fractions of all baskets):

* support      = P(A and C)
* confidence   = P(C | A)            = support(A and C) / support(A)
* lift         = confidence / P(C)   = how much more often C is bought with A
                 than expected if A and C were independent (1.0 = independent)
* leverage     = support(A and C) - support(A) * support(C)
* conviction   = (1 - support(C)) / (1 - confidence)   (inf when confidence = 1)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth

ALGORITHMS = ("fpgrowth", "apriori")
ITEM_SEP = " | "  # separator used when itemsets are written to CSV

ITEMSET_COLUMNS = ["support", "itemsets", "length"]
RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedent_support",
    "consequent_support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]


def mine_frequent_itemsets(
    onehot: pd.DataFrame,
    min_support: float = 0.01,
    algorithm: str = "fpgrowth",
    max_len: int | None = 4,
) -> pd.DataFrame:
    """Find all itemsets whose support is at least ``min_support``.

    Returns columns ``support``, ``itemsets`` (frozenset) and ``length``,
    sorted by support (highest first).
    """
    if not 0.0 < min_support <= 1.0:
        raise ValueError("min_support must be in the interval (0, 1]")
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got '{algorithm}'")
    if max_len is not None and max_len < 1:
        raise ValueError("max_len must be >= 1")

    if onehot.shape[1] == 0 or onehot.shape[0] == 0:
        return pd.DataFrame({"support": [], "itemsets": [], "length": []}, columns=ITEMSET_COLUMNS)

    miner = fpgrowth if algorithm == "fpgrowth" else apriori
    itemsets = miner(onehot, min_support=min_support, use_colnames=True, max_len=max_len)

    if itemsets.empty:
        return pd.DataFrame({"support": [], "itemsets": [], "length": []}, columns=ITEMSET_COLUMNS)

    itemsets = itemsets.copy()
    itemsets["length"] = itemsets["itemsets"].apply(len)
    itemsets = itemsets.sort_values(
        ["support", "length"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return itemsets[ITEMSET_COLUMNS]


def generate_rules(
    itemsets: pd.DataFrame,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Generate association rules from frequent itemsets.

    Every frequent itemset of size >= 2 is split in all possible ways into a
    non-empty antecedent and consequent. Rules are kept if their confidence is
    at least ``min_confidence`` and their lift is at least ``min_lift``.

    Returns rules sorted by lift, then confidence, then support (descending).
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    if min_lift < 0:
        raise ValueError("min_lift must be >= 0")

    support_of = {fs: float(s) for fs, s in zip(itemsets["itemsets"], itemsets["support"])}

    records = []
    for itemset, support in support_of.items():
        if len(itemset) < 2:
            continue
        members = sorted(itemset)
        for size in range(1, len(members)):
            for antecedent_items in combinations(members, size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                # Subsets of a frequent itemset are always frequent, so both
                # lookups succeed; the guard only protects against odd input.
                ante_support = support_of.get(antecedent)
                cons_support = support_of.get(consequent)
                if not ante_support or not cons_support:
                    continue

                confidence = support / ante_support
                lift = confidence / cons_support
                if confidence < min_confidence - 1e-12 or lift < min_lift - 1e-12:
                    continue

                conviction = np.inf if confidence >= 1.0 - 1e-12 else (1.0 - cons_support) / (1.0 - confidence)
                records.append(
                    {
                        "antecedents": antecedent,
                        "consequents": consequent,
                        "antecedent_support": ante_support,
                        "consequent_support": cons_support,
                        "support": support,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": support - ante_support * cons_support,
                        "conviction": conviction,
                    }
                )

    if not records:
        return pd.DataFrame(columns=RULE_COLUMNS)

    rules = pd.DataFrame.from_records(records, columns=RULE_COLUMNS)
    # Tie-break on the item names so ordering is fully deterministic.
    rules["_a"] = rules["antecedents"].apply(format_itemset)
    rules["_c"] = rules["consequents"].apply(format_itemset)
    rules = rules.sort_values(
        ["lift", "confidence", "support", "_a", "_c"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return rules.drop(columns=["_a", "_c"]).reset_index(drop=True)


def prune_redundant_rules(rules: pd.DataFrame, min_confidence_gain: float = 0.05) -> pd.DataFrame:
    """Drop rules whose extra antecedent items do not pay for themselves.

    A rule such as ``Dog Food + Milk -> Dog Treats`` is redundant when a more
    general rule with the same consequent (``Dog Food -> Dog Treats``) is
    almost as confident: the extra item adds complexity, not information.
    A rule is kept only if its confidence beats that of *every* more general
    rule (same consequent, antecedent a proper subset) by at least
    ``min_confidence_gain`` (0.05 = five percentage points). Rules with no
    more general counterpart are always kept. Order is preserved.
    """
    if min_confidence_gain < 0:
        raise ValueError("min_confidence_gain must be >= 0")
    if rules.empty:
        return rules.copy()

    antecedents = rules["antecedents"].tolist()
    confidence = rules["confidence"].to_numpy(dtype=float)

    by_consequent: dict[frozenset, list[int]] = {}
    for position, consequent in enumerate(rules["consequents"]):
        by_consequent.setdefault(consequent, []).append(position)

    keep = np.ones(len(rules), dtype=bool)
    for group in by_consequent.values():
        if len(group) < 2:
            continue
        for i in group:
            for j in group:
                if antecedents[j] < antecedents[i] and confidence[i] - confidence[j] < min_confidence_gain - 1e-12:
                    keep[i] = False
                    break
    return rules[keep].reset_index(drop=True)


def top_rules(
    rules: pd.DataFrame,
    n: int = 10,
    one_per_itemset: bool = True,
    prune_redundant: bool = True,
) -> pd.DataFrame:
    """The ``n`` strongest rules by lift (ties broken by confidence).

    This is a curated view for reports; the full rule set stays untouched.

    * ``prune_redundant``: drop rules made redundant by a simpler rule
      (see :func:`prune_redundant_rules`).
    * ``one_per_itemset``: one itemset yields many rules (``A -> B``,
      ``B -> A``, ``A -> B + C`` ...) that describe the same co-purchase, so
      keep only the best rule per itemset.
    """
    if rules.empty:
        return rules.copy()
    ranked = rules.sort_values(["lift", "confidence"], ascending=False, kind="mergesort").reset_index(drop=True)
    if prune_redundant:
        ranked = prune_redundant_rules(ranked)
    if one_per_itemset and not ranked.empty:
        union = ranked.apply(lambda r: r["antecedents"] | r["consequents"], axis=1)
        ranked = ranked[~union.duplicated(keep="first")]
    return ranked.head(n).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Formatting and persistence
# --------------------------------------------------------------------------- #
def format_itemset(itemset) -> str:
    """Stable text form of an itemset: sorted items joined with ``ITEM_SEP``."""
    return ITEM_SEP.join(sorted(itemset))


def parse_itemset(text: str) -> frozenset:
    """Inverse of :func:`format_itemset`."""
    return frozenset(part for part in str(text).split(ITEM_SEP) if part != "")


def rule_label(antecedents, consequents, arrow: str = "->") -> str:
    """Readable label such as ``Butter + Jam -> Bread``."""
    left = " + ".join(sorted(antecedents))
    right = " + ".join(sorted(consequents))
    return f"{left} {arrow} {right}"


def save_rules(rules: pd.DataFrame, path: str | Path) -> Path:
    """Write rules to CSV with itemsets as text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(format_itemset)
    out["consequents"] = out["consequents"].apply(format_itemset)
    out.to_csv(path, index=False, encoding="utf-8")
    return path


def load_rules(path: str | Path) -> pd.DataFrame:
    """Read rules written by :func:`save_rules` back into frozenset form."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rules file not found: {path}")
    rules = pd.read_csv(path, encoding="utf-8")
    missing = set(RULE_COLUMNS) - set(rules.columns)
    if missing:
        raise ValueError(f"Rules file is missing columns: {sorted(missing)}")
    rules["antecedents"] = rules["antecedents"].apply(parse_itemset)
    rules["consequents"] = rules["consequents"].apply(parse_itemset)
    return rules


def save_itemsets(itemsets: pd.DataFrame, path: str | Path) -> Path:
    """Write frequent itemsets to CSV with itemsets as text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = itemsets.copy()
    out["itemsets"] = out["itemsets"].apply(format_itemset)
    out.to_csv(path, index=False, encoding="utf-8")
    return path
