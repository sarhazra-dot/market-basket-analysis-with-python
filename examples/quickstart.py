"""Use market basket analysis as a library instead of via the CLI.

Run from anywhere:  python examples/quickstart.py
"""

from pathlib import Path

from marketbasket import (
    build_baskets,
    clean_transactions,
    encode_baskets,
    generate_rules,
    load_transactions,
    mine_frequent_itemsets,
    recommend,
)
from marketbasket.rules import rule_label, top_rules

DATA = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

# 1. Load and clean (drops cancellations, postage rows, bad quantities, ...)
raw = load_transactions(DATA)
clean = clean_transactions(raw, min_basket_size=2)

# 2. One list of items per basket, then a True/False matrix (rows = baskets)
baskets = build_baskets(clean)
onehot = encode_baskets(baskets, min_item_support=0.01)

# 3. Frequent itemsets -> association rules
itemsets = mine_frequent_itemsets(onehot, min_support=0.01, max_len=3)
rules = generate_rules(itemsets, min_confidence=0.4, min_lift=2.0)
print(f"{len(baskets):,} baskets -> {len(itemsets):,} itemsets -> {len(rules):,} rules\n")

# 4. Strongest findings (redundant rules removed, one rule per itemset)
print("Top rules:")
for rule in top_rules(rules, n=5).itertuples():
    print(f"  {rule_label(rule.antecedents, rule.consequents)}  (lift {rule.lift:.1f}, confidence {rule.confidence:.2f})")

# 5. "Customers also bought" for a basket
print("\nCustomer has Spaghetti in the basket. Suggest:")
print(recommend(rules, ["Spaghetti"], top_n=3)[["item", "confidence", "lift"]].to_string(index=False))
