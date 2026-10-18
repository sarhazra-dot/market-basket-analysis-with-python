"""End-to-end pipeline: load -> clean -> baskets -> mine -> rules -> outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .data import clean_transactions, load_transactions
from .preprocess import build_baskets, encode_baskets
from .report import write_summary
from .rules import generate_rules, mine_frequent_itemsets, save_itemsets, save_rules
from .viz import plot_lift_heatmap, plot_support_confidence, plot_top_items, plot_top_rules

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All settings for one analysis run."""

    input_path: str | Path
    output_dir: str | Path = "outputs"
    min_support: float = 0.01
    min_confidence: float = 0.3
    min_lift: float = 1.0
    max_len: int | None = 4
    algorithm: str = "fpgrowth"
    min_basket_size: int = 2
    invoice_col: str | None = None
    item_col: str | None = None
    quantity_col: str | None = None
    price_col: str | None = None
    top_n_plots: int = 15


@dataclass
class PipelineResult:
    """What a run produced."""

    stats: dict
    itemsets: pd.DataFrame
    rules: pd.DataFrame
    files: list[Path] = field(default_factory=list)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full analysis and write results to ``config.output_dir``."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    logger.info("Loading %s", config.input_path)
    raw = load_transactions(
        config.input_path,
        invoice_col=config.invoice_col,
        item_col=config.item_col,
        quantity_col=config.quantity_col,
        price_col=config.price_col,
    )

    logger.info("Cleaning %d rows", len(raw))
    clean = clean_transactions(raw, min_basket_size=config.min_basket_size)
    if clean.empty:
        raise ValueError(
            "No transactions left after cleaning. Check the column mapping and "
            "that baskets contain at least --min-basket-size items."
        )

    baskets = build_baskets(clean)
    onehot = encode_baskets(baskets, min_item_support=config.min_support)

    logger.info("Mining itemsets (%s, min_support=%s)", config.algorithm, config.min_support)
    itemsets = mine_frequent_itemsets(
        onehot,
        min_support=config.min_support,
        algorithm=config.algorithm,
        max_len=config.max_len,
    )
    rules = generate_rules(
        itemsets,
        min_confidence=config.min_confidence,
        min_lift=config.min_lift,
    )

    stats = {
        "n_rows_raw": int(len(raw)),
        "n_rows_clean": int(len(clean)),
        "n_baskets": int(len(baskets)),
        "n_items": int(clean["item"].nunique()),
        "avg_basket_size": float(sum(len(b) for b in baskets) / len(baskets)),
        "n_itemsets": int(len(itemsets)),
        "n_rules": int(len(rules)),
    }
    params = {
        "algorithm": config.algorithm,
        "min_support": config.min_support,
        "min_confidence": config.min_confidence,
        "min_lift": config.min_lift,
        "max_len": config.max_len,
        "min_basket_size": config.min_basket_size,
    }

    files.append(save_itemsets(itemsets, out_dir / "frequent_itemsets.csv"))
    files.append(save_rules(rules, out_dir / "rules.csv"))

    if onehot.shape[1] > 0:
        files.append(plot_top_items(onehot, out_dir / "top_items.png", top_n=config.top_n_plots))
    if not rules.empty:
        files.append(plot_support_confidence(rules, out_dir / "support_confidence.png"))
        files.append(plot_top_rules(rules, out_dir / "top_rules.png", top_n=config.top_n_plots))
        heatmap = plot_lift_heatmap(rules, out_dir / "lift_heatmap.png")
        if heatmap is not None:
            files.append(heatmap)

    files.append(write_summary(out_dir / "summary.md", stats, params, rules))
    logger.info("Wrote %d files to %s", len(files), out_dir)
    return PipelineResult(stats=stats, itemsets=itemsets, rules=rules, files=files)
