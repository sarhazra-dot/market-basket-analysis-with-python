"""Command-line interface: ``marketbasket`` or ``python -m marketbasket``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .data import generate_synthetic_transactions
from .pipeline import PipelineConfig, run_pipeline
from .recommend import known_items, recommend
from .rules import ALGORITHMS, load_rules, rule_label, top_rules


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marketbasket",
        description="Market basket analysis: mine association rules from retail transactions.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Create a reproducible synthetic dataset")
    gen.add_argument("--out", default="data/sample_transactions.csv", help="Output CSV path")
    gen.add_argument("--n-transactions", type=int, default=3000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--no-noise", action="store_true", help="Omit cancellations, postage and missing values")

    run = sub.add_parser("run", help="Run the full analysis on a transactions file")
    run.add_argument("--input", required=True, help="CSV or XLSX file of transactions (one row per item)")
    run.add_argument("--output-dir", default="outputs")
    run.add_argument("--min-support", type=float, default=0.01)
    run.add_argument("--min-confidence", type=float, default=0.3)
    run.add_argument("--min-lift", type=float, default=1.0)
    run.add_argument("--max-len", type=int, default=4, help="Maximum itemset size")
    run.add_argument("--algorithm", choices=ALGORITHMS, default="fpgrowth")
    run.add_argument("--min-basket-size", type=int, default=2)
    run.add_argument("--invoice-col", help="Basket id column (auto-detected if omitted)")
    run.add_argument("--item-col", help="Item column (auto-detected if omitted)")
    run.add_argument("--quantity-col", help="Quantity column (auto-detected if omitted)")
    run.add_argument("--price-col", help="Unit price column (auto-detected if omitted)")
    run.add_argument("--top-n", type=int, default=15, help="Rows shown in the bar charts")

    rec = sub.add_parser("recommend", help="Suggest items to add to a basket")
    rec.add_argument("--rules", default="outputs/rules.csv", help="rules.csv produced by 'run'")
    rec.add_argument(
        "--item", action="append", required=True, help="Item already in the basket (repeat for several)"
    )
    rec.add_argument("--top-n", type=int, default=5)

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    df = generate_synthetic_transactions(
        n_transactions=args.n_transactions,
        seed=args.seed,
        include_noise=not args.no_noise,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df):,} rows ({args.n_transactions:,} transactions) to {out}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_len=args.max_len,
        algorithm=args.algorithm,
        min_basket_size=args.min_basket_size,
        invoice_col=args.invoice_col,
        item_col=args.item_col,
        quantity_col=args.quantity_col,
        price_col=args.price_col,
        top_n_plots=args.top_n,
    )
    result = run_pipeline(config)
    s = result.stats
    print(
        f"Baskets: {s['n_baskets']:,} | items: {s['n_items']:,} | "
        f"frequent itemsets: {s['n_itemsets']:,} | rules: {s['n_rules']:,}"
    )
    if result.rules.empty:
        print("No rules met the thresholds. Try lowering --min-support or --min-confidence.")
    else:
        print("\nTop 5 rules by lift (redundant rules removed):")
        for row in top_rules(result.rules, n=5).itertuples(index=False):
            label = rule_label(row.antecedents, row.consequents)
            print(f"  {label}  (support={row.support:.3f}, confidence={row.confidence:.2f}, lift={row.lift:.2f})")
    print(f"\nResults written to {Path(args.output_dir)}/")
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    rules = load_rules(args.rules)
    vocabulary = {name.casefold() for name in known_items(rules)}
    unknown = [i for i in args.item if i.strip().casefold() not in vocabulary]
    if unknown:
        print(f"Note: not found in the rules and ignored: {', '.join(unknown)}", file=sys.stderr)

    result = recommend(rules, args.item, top_n=args.top_n)
    if result.empty:
        print("No recommendations for this basket.")
        return 0

    print(f"Basket: {', '.join(args.item)}\nRecommended additions:")
    for row in result.itertuples(index=False):
        print(
            f"  {row.item}  (because of: {row.because}; "
            f"confidence={row.confidence:.2f}, lift={row.lift:.2f})"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    handlers = {"generate-data": _cmd_generate, "run": _cmd_run, "recommend": _cmd_recommend}
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
