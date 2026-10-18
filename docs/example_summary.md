# Market Basket Analysis Summary

## Dataset

- Raw rows: 14,252
- Rows after cleaning: 13,968
- Baskets (transactions): 2,949
- Distinct items: 68
- Average basket size: 4.74 items

## Parameters

- `algorithm`: fpgrowth
- `min_support`: 0.01
- `min_confidence`: 0.3
- `min_lift`: 1.0
- `max_len`: 4
- `min_basket_size`: 2

## Results

- Frequent itemsets found: 422
- Association rules kept: 752

## Top 10 rules by lift

Redundant rules (extra items that add little confidence) are removed and only the best rule per itemset is shown; the full list is in `rules.csv`.

| Rule | Support | Confidence | Lift |
|---|---:|---:|---:|
| Baby Formula → Baby Wipes | 0.027 | 0.870 | 14.74 |
| Baby Wipes → Baby Formula + Diapers | 0.025 | 0.425 | 14.58 |
| Dog Food + Milk → Dog Treats | 0.013 | 0.520 | 14.33 |
| Baby Wipes → Diapers | 0.056 | 0.948 | 14.05 |
| Diapers → Baby Formula | 0.029 | 0.432 | 13.85 |
| Beef Patties + Milk → Burger Buns + Cheddar Slices | 0.013 | 0.534 | 12.91 |
| Beef Patties + Milk → Beer + Burger Buns | 0.011 | 0.425 | 12.91 |
| Almonds → Blueberries + Granola | 0.010 | 0.366 | 12.84 |
| Almonds → Blueberries + Greek Yogurt | 0.011 | 0.402 | 12.76 |
| Almonds → Blueberries | 0.014 | 0.500 | 12.71 |

## Reading the metrics

- **Support**: share of baskets containing every item in the rule.
- **Confidence**: of the baskets containing the left side, the share that also contain the right side.
- **Lift**: confidence divided by how common the right side is overall. Above 1 means the items are bought together more often than chance; 1 means no relationship.

Association is not causation: a rule says items co-occur, not that one purchase causes the other.
