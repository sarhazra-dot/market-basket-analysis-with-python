"""Write a human-readable Markdown summary of an analysis run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .rules import rule_label, top_rules


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_summary(
    path: str | Path,
    stats: dict,
    params: dict,
    rules: pd.DataFrame,
    top_n: int = 10,
) -> Path:
    """Write ``summary.md`` with dataset stats, parameters and the top rules."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Market Basket Analysis Summary",
        "",
        "## Dataset",
        "",
        f"- Raw rows: {stats['n_rows_raw']:,}",
        f"- Rows after cleaning: {stats['n_rows_clean']:,}",
        f"- Baskets (transactions): {stats['n_baskets']:,}",
        f"- Distinct items: {stats['n_items']:,}",
        f"- Average basket size: {stats['avg_basket_size']:.2f} items",
        "",
        "## Parameters",
        "",
    ]
    lines += [f"- `{key}`: {value}" for key, value in params.items()]
    lines += [
        "",
        "## Results",
        "",
        f"- Frequent itemsets found: {stats['n_itemsets']:,}",
        f"- Association rules kept: {stats['n_rules']:,}",
        "",
    ]

    if rules.empty:
        lines += [
            "No rules met the thresholds. Try lowering `min_support` or `min_confidence`.",
            "",
        ]
    else:
        top = top_rules(rules, n=top_n)
        lines += [
            f"## Top {len(top)} rules by lift",
            "",
            "Redundant rules (extra items that add little confidence) are removed and only the best rule per itemset is shown; the full list is in `rules.csv`.",
            "",
            "| Rule | Support | Confidence | Lift |",
            "|---|---:|---:|---:|",
        ]
        for row in top.itertuples(index=False):
            label = _escape(rule_label(row.antecedents, row.consequents, arrow="\u2192"))
            lines.append(f"| {label} | {row.support:.3f} | {row.confidence:.3f} | {row.lift:.2f} |")
        lines.append("")

    lines += [
        "## Reading the metrics",
        "",
        "- **Support**: share of baskets containing every item in the rule.",
        "- **Confidence**: of the baskets containing the left side, the share that also contain the right side.",
        "- **Lift**: confidence divided by how common the right side is overall. "
        "Above 1 means the items are bought together more often than chance; 1 means no relationship.",
        "",
        "Association is not causation: a rule says items co-occur, not that one purchase causes the other.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
