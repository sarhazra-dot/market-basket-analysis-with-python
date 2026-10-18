"""Market basket analysis toolkit."""

__version__ = "1.0.0"

from .data import clean_transactions, generate_synthetic_transactions, load_transactions
from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .preprocess import build_baskets, encode_baskets
from .recommend import recommend
from .rules import generate_rules, mine_frequent_itemsets

__all__ = [
    "__version__",
    "PipelineConfig",
    "PipelineResult",
    "build_baskets",
    "clean_transactions",
    "encode_baskets",
    "generate_rules",
    "generate_synthetic_transactions",
    "load_transactions",
    "mine_frequent_itemsets",
    "recommend",
    "run_pipeline",
]
