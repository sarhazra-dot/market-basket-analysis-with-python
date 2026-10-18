"""Plots for the analysis. Uses matplotlib's object-oriented API only, so no
display or global backend is needed (works on servers and in CI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .rules import rule_label, top_rules


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def plot_top_items(onehot: pd.DataFrame, path: str | Path, top_n: int = 15) -> Path:
    """Bar chart of the most frequently bought items (share of baskets)."""
    freq = onehot.mean().sort_values(ascending=False).head(top_n)[::-1]
    fig = Figure(figsize=(8, max(4.0, 0.38 * len(freq) + 1.5)))
    ax = fig.subplots()
    ax.barh(list(freq.index), freq.values * 100, color="#4C78A8")
    ax.set_xlabel("Share of baskets containing the item (%)")
    ax.set_title(f"Top {len(freq)} items by popularity")
    for y, value in enumerate(freq.values * 100):
        ax.text(value, y, f" {value:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, max(freq.values * 100) * 1.15)
    return _save(fig, path)


def plot_support_confidence(rules: pd.DataFrame, path: str | Path) -> Path:
    """Scatter plot of support vs confidence, coloured by lift."""
    fig = Figure(figsize=(8, 6))
    ax = fig.subplots()
    points = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        s=34,
        alpha=0.8,
        edgecolors="none",
    )
    fig.colorbar(points, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Association rules ({len(rules)} rules)")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_top_rules(rules: pd.DataFrame, path: str | Path, top_n: int = 15) -> Path:
    """Horizontal bar chart of the rules with the highest lift (one per itemset)."""
    top = top_rules(rules, n=top_n).iloc[::-1]
    labels = [rule_label(a, c, arrow="\u2192") for a, c in zip(top["antecedents"], top["consequents"])]
    fig = Figure(figsize=(10, max(4.0, 0.42 * len(top) + 1.5)))
    ax = fig.subplots()
    ax.barh(labels, top["lift"].values, color="#F58518")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Lift (1.0 = no association)")
    ax.set_title(f"Top {len(top)} rules by lift")
    for y, value in enumerate(top["lift"].values):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, float(top["lift"].max()) * 1.15)
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, path)


def plot_lift_heatmap(rules: pd.DataFrame, path: str | Path, top_n: int = 12) -> Path | None:
    """Heatmap of lift between single items (rows: antecedent, columns: consequent).

    Returns ``None`` when there are no item-to-item rules to show.
    """
    pairs = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)]
    pairs = pairs.sort_values(["lift", "confidence"], ascending=False)
    if pairs.empty:
        return None

    items: list[str] = []
    for a, c in zip(pairs["antecedents"], pairs["consequents"]):
        for item in (next(iter(a)), next(iter(c))):
            if item not in items:
                items.append(item)
        if len(items) >= top_n:
            break
    items = sorted(items[:top_n])
    index = {item: i for i, item in enumerate(items)}

    matrix = np.full((len(items), len(items)), np.nan)
    for a, c, lift in zip(pairs["antecedents"], pairs["consequents"], pairs["lift"]):
        ai, ci = next(iter(a)), next(iter(c))
        if ai in index and ci in index:
            matrix[index[ai], index[ci]] = lift

    size = max(6.0, 0.62 * len(items) + 2.5)
    fig = Figure(figsize=(size, size * 0.9))
    ax = fig.subplots()
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="YlOrRd", aspect="auto")
    fig.colorbar(image, ax=ax, label="Lift")
    ax.set_xticks(range(len(items)))
    ax.set_yticks(range(len(items)))
    ax.set_xticklabels(items, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(items, fontsize=8)
    ax.set_xlabel("Consequent (bought with)")
    ax.set_ylabel("Antecedent (already in basket)")
    ax.set_title("Lift between items")
    for i in range(len(items)):
        for j in range(len(items)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    return _save(fig, path)
