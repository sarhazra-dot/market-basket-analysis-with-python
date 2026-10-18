"""Loading, cleaning and generating transaction data.

The pipeline works on *long-format* transaction data: one row per item sold,
with a column identifying the basket (invoice / order / transaction id) and a
column naming the item. The default column aliases match the public
"Online Retail" / "Online Retail II" datasets, so those load without any
configuration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column-name aliases (compared case-insensitively, spaces ignored).
INVOICE_ALIASES = (
    "invoiceno",
    "invoice",
    "invoice_no",
    "invoice_id",
    "order_id",
    "transaction_id",
    "basket_id",
)
ITEM_ALIASES = ("description", "item", "product", "product_name", "item_name")
QUANTITY_ALIASES = ("quantity", "qty")
PRICE_ALIASES = ("unitprice", "price", "unit_price")

# Rows in retail datasets that are not real products.
NON_PRODUCT_ITEMS = frozenset(
    {
        "POSTAGE",
        "DOTCOM POSTAGE",
        "MANUAL",
        "BANK CHARGES",
        "CARRIAGE",
        "DISCOUNT",
        "SAMPLES",
        "AMAZON FEE",
        "ADJUST BAD DEBT",
        "CRUK COMMISSION",
        "POST",
        "DOT",
        "AMAZONFEE",
        "CRUK",
        "C2",
    }
)

# Cancelled invoices in the Online Retail datasets look like "C536379".
_CANCELLATION_PATTERN = r"^C\d"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _normalize(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def _find_column(
    columns: list[str],
    aliases: tuple[str, ...],
    explicit: str | None,
    label: str,
    required: bool,
) -> str | None:
    """Return the actual column name to use for a logical column."""
    if explicit is not None:
        if explicit not in columns:
            raise ValueError(
                f"{label} column '{explicit}' not found. Available columns: {columns}"
            )
        return explicit

    normalized = {_normalize(c): c for c in columns}
    for alias in aliases:
        if _normalize(alias) in normalized:
            return normalized[_normalize(alias)]

    if required:
        raise ValueError(
            f"Could not detect the {label} column automatically. "
            f"Available columns: {columns}. Pass it explicitly "
            f"(e.g. --{label}-col NAME)."
        )
    return None


def load_transactions(
    path: str | Path,
    invoice_col: str | None = None,
    item_col: str | None = None,
    quantity_col: str | None = None,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Load a transactions file (.csv or .xlsx) into a standardized DataFrame.

    The returned frame has the columns ``invoice`` and ``item`` plus, when
    available, ``quantity`` and ``price``. All values are read as text and
    quantity / price are converted to numbers (invalid values become NaN).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            # Read every sheet: Online Retail II stores each year on its own sheet.
            sheets = pd.read_excel(path, dtype=str, sheet_name=None)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Reading Excel files needs openpyxl: pip install openpyxl"
            ) from exc
        raw = pd.concat(sheets.values(), ignore_index=True)
    else:
        try:
            raw = pd.read_csv(path, dtype=str)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, dtype=str, encoding="latin-1")

    columns = list(raw.columns)
    inv = _find_column(columns, INVOICE_ALIASES, invoice_col, "invoice", True)
    itm = _find_column(columns, ITEM_ALIASES, item_col, "item", True)
    qty = _find_column(columns, QUANTITY_ALIASES, quantity_col, "quantity", False)
    prc = _find_column(columns, PRICE_ALIASES, price_col, "price", False)

    out = pd.DataFrame({"invoice": raw[inv], "item": raw[itm]})
    if qty is not None:
        out["quantity"] = pd.to_numeric(raw[qty], errors="coerce")
    if prc is not None:
        out["price"] = pd.to_numeric(raw[prc], errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_transactions(
    df: pd.DataFrame,
    min_basket_size: int = 2,
    drop_cancellations: bool = True,
    drop_non_products: bool = True,
) -> pd.DataFrame:
    """Clean standardized transaction rows.

    Steps: drop missing ids/items, tidy whitespace, drop cancelled invoices,
    drop rows with non-positive quantity or price (when those columns exist),
    drop non-product rows (postage, fees, ...), de-duplicate items within a
    basket, and drop baskets smaller than ``min_basket_size`` (single-item
    baskets cannot produce association rules).

    Returns a frame with exactly two columns: ``invoice`` and ``item``.
    """
    if min_basket_size < 1:
        raise ValueError("min_basket_size must be >= 1")
    missing = {"invoice", "item"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.dropna(subset=["invoice", "item"]).copy()
    out["invoice"] = out["invoice"].astype(str).str.strip()
    out["item"] = out["item"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    out = out[(out["invoice"] != "") & (out["item"] != "")]

    if drop_cancellations:
        is_cancel = out["invoice"].str.contains(_CANCELLATION_PATTERN, case=False, regex=True)
        out = out[~is_cancel]

    if "quantity" in out.columns:
        out = out[out["quantity"] > 0]  # NaN compares False, so invalid rows drop too
    if "price" in out.columns:
        out = out[out["price"] > 0]

    if drop_non_products:
        out = out[~out["item"].str.upper().isin(NON_PRODUCT_ITEMS)]

    out = out[["invoice", "item"]].drop_duplicates()

    basket_size = out.groupby("invoice")["item"].transform("size")
    out = out[basket_size >= min_basket_size]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
# Shopping "missions". Each mission is (name, weight, [(item, probability)]).
# Items from the same mission tend to be bought together, which plants real
# associations for the algorithm to rediscover.
PATTERNS: tuple[tuple[str, float, tuple[tuple[str, float], ...]], ...] = (
    (
        "Breakfast",
        1.4,
        (("Whole Wheat Bread", 0.90), ("Butter", 0.80), ("Strawberry Jam", 0.50), ("Orange Juice", 0.35)),
    ),
    (
        "Pasta Night",
        1.3,
        (("Spaghetti", 0.95), ("Tomato Sauce", 0.85), ("Parmesan Cheese", 0.60), ("Garlic Bread", 0.30)),
    ),
    (
        "Movie Night",
        1.2,
        (("Potato Chips", 0.90), ("Cola", 0.80), ("Popcorn", 0.50), ("Chocolate Bar", 0.30)),
    ),
    (
        "Baking",
        1.0,
        (("Flour", 0.90), ("Sugar", 0.80), ("Eggs", 0.80), ("Vanilla Extract", 0.50), ("Baking Powder", 0.45)),
    ),
    (
        "Barbecue",
        0.9,
        (("Burger Buns", 0.90), ("Beef Patties", 0.90), ("Ketchup", 0.60), ("Cheddar Slices", 0.50), ("Beer", 0.40)),
    ),
    (
        "Baby Care",
        0.8,
        (("Diapers", 0.95), ("Baby Wipes", 0.80), ("Baby Formula", 0.40)),
    ),
    (
        "Coffee Break",
        1.0,
        (("Ground Coffee", 0.90), ("Coffee Filters", 0.60), ("Coffee Creamer", 0.50)),
    ),
    (
        "Healthy Snack",
        0.9,
        (("Greek Yogurt", 0.80), ("Granola", 0.70), ("Blueberries", 0.50), ("Almonds", 0.30)),
    ),
    (
        "Salad",
        0.9,
        (("Lettuce", 0.90), ("Tomatoes", 0.70), ("Cucumber", 0.60), ("Olive Oil", 0.40)),
    ),
    (
        "Pet Care",
        0.7,
        (("Dog Food", 0.90), ("Dog Treats", 0.50)),
    ),
)

# Everyday items bought independently of any mission, most popular first.
NOISE_ITEMS: tuple[str, ...] = (
    "Milk",
    "Bananas",
    "Bottled Water",
    "Apples",
    "Rice",
    "Onions",
    "Chicken Breast",
    "Potatoes",
    "Toilet Paper",
    "Carrots",
    "Cereal",
    "Frozen Pizza",
    "Paper Towels",
    "Dish Soap",
    "Tea Bags",
    "Peanut Butter",
    "Canned Beans",
    "Ice Cream",
    "Laundry Detergent",
    "Shampoo",
    "Toothpaste",
    "Orange",
    "Hand Soap",
    "Black Pepper",
    "Salt",
    "Honey",
    "Canned Tuna",
    "Mayonnaise",
    "Mustard",
    "Trash Bags",
)

PRICES: dict[str, float] = {
    "Whole Wheat Bread": 2.5, "Butter": 3.2, "Strawberry Jam": 2.8, "Orange Juice": 3.5,
    "Spaghetti": 1.8, "Tomato Sauce": 2.2, "Parmesan Cheese": 4.5, "Garlic Bread": 2.9,
    "Potato Chips": 2.4, "Cola": 1.9, "Popcorn": 2.1, "Chocolate Bar": 1.5,
    "Flour": 2.0, "Sugar": 1.7, "Eggs": 3.0, "Vanilla Extract": 5.5, "Baking Powder": 1.9,
    "Burger Buns": 2.6, "Beef Patties": 6.8, "Ketchup": 2.3, "Cheddar Slices": 3.9, "Beer": 8.5,
    "Diapers": 12.0, "Baby Wipes": 4.0, "Baby Formula": 18.0,
    "Ground Coffee": 7.5, "Coffee Filters": 2.2, "Coffee Creamer": 3.1,
    "Greek Yogurt": 1.4, "Granola": 4.2, "Blueberries": 4.8, "Almonds": 6.0,
    "Lettuce": 1.6, "Tomatoes": 2.7, "Cucumber": 1.1, "Olive Oil": 9.0,
    "Dog Food": 14.0, "Dog Treats": 5.0,
    "Milk": 1.2, "Bananas": 0.6, "Bottled Water": 0.9, "Apples": 2.1, "Rice": 3.4,
    "Onions": 1.3, "Chicken Breast": 7.2, "Potatoes": 2.4, "Toilet Paper": 6.5, "Carrots": 1.2,
    "Cereal": 3.8, "Frozen Pizza": 5.5, "Paper Towels": 4.6, "Dish Soap": 2.5, "Tea Bags": 3.3,
    "Peanut Butter": 3.7, "Canned Beans": 1.3, "Ice Cream": 4.9, "Laundry Detergent": 11.0,
    "Shampoo": 5.2, "Toothpaste": 2.9, "Orange": 0.8, "Hand Soap": 2.1, "Black Pepper": 2.6,
    "Salt": 0.9, "Honey": 5.8, "Canned Tuna": 1.8, "Mayonnaise": 3.0, "Mustard": 1.9,
    "Trash Bags": 4.1,
}

_COUNTRIES = ("India", "United Kingdom", "Germany", "France", "United States")
_COUNTRY_WEIGHTS = (0.40, 0.25, 0.15, 0.10, 0.10)


def generate_synthetic_transactions(
    n_transactions: int = 3000,
    seed: int = 42,
    include_noise: bool = True,
) -> pd.DataFrame:
    """Generate a reproducible retail dataset in the Online Retail schema.

    Columns: InvoiceNo, InvoiceDate, Description, Quantity, UnitPrice,
    CustomerID, Country.

    Baskets are built from shopping "missions" (breakfast, pasta night, ...)
    plus independently chosen everyday items, so the data contains known
    associations. With ``include_noise=True`` the output also contains rows a
    real dataset would: cancelled invoices (negative quantity), postage rows
    and a few missing descriptions. The cleaning step is expected to remove
    them.
    """
    if n_transactions < 1:
        raise ValueError("n_transactions must be >= 1")

    rng = np.random.default_rng(seed)

    pattern_weights = np.array([p[1] for p in PATTERNS], dtype=float)
    pattern_weights /= pattern_weights.sum()

    noise_weights = 1.0 / np.arange(1, len(NOISE_ITEMS) + 1) ** 0.8
    noise_weights /= noise_weights.sum()

    start = np.datetime64("2023-01-01T00:00:00")
    minutes_in_year = 365 * 24 * 60

    rows: list[tuple] = []
    baskets: list[list[str]] = []

    for idx in range(n_transactions):
        invoice = f"T{idx + 1:06d}"
        items: set[str] = set()

        use_missions = rng.random() >= 0.15
        if use_missions:
            n_missions = int(rng.choice([1, 2], p=[0.75, 0.25]))
            chosen = rng.choice(len(PATTERNS), size=n_missions, replace=False, p=pattern_weights)
            for pattern_idx in chosen:
                for item, prob in PATTERNS[int(pattern_idx)][2]:
                    if rng.random() < prob:
                        items.add(item)

        n_noise = int(rng.poisson(2.0))
        if not use_missions:
            n_noise = max(n_noise, 2)
        n_noise = min(n_noise, len(NOISE_ITEMS))
        if n_noise > 0:
            picked = rng.choice(len(NOISE_ITEMS), size=n_noise, replace=False, p=noise_weights)
            items.update(NOISE_ITEMS[int(i)] for i in picked)
        if not items:
            items.add(NOISE_ITEMS[int(rng.choice(len(NOISE_ITEMS), p=noise_weights))])

        when = start + np.timedelta64(int(rng.integers(0, minutes_in_year)), "m")
        timestamp = str(when).replace("T", " ")[:19]
        customer = str(int(rng.integers(10000, 10600)))
        country = str(rng.choice(_COUNTRIES, p=_COUNTRY_WEIGHTS))

        ordered = sorted(items)
        baskets.append(ordered)
        for item in ordered:
            quantity = int(rng.integers(1, 4))
            rows.append((invoice, timestamp, item, quantity, PRICES[item], customer, country))

        if include_noise and rng.random() < 0.03:
            rows.append((invoice, timestamp, "POSTAGE", 1, 15.0, customer, country))

    if include_noise:
        n_cancel = max(1, int(0.02 * n_transactions))
        cancel_ids = rng.choice(n_transactions, size=min(n_cancel, n_transactions), replace=False)
        for idx in cancel_ids:
            idx = int(idx)
            returned = baskets[idx][: int(rng.integers(1, 3))]
            ts = str(start + np.timedelta64(int(rng.integers(0, minutes_in_year)), "m")).replace("T", " ")[:19]
            for item in returned:
                rows.append((f"C{idx + 1:06d}", ts, item, -1, PRICES[item], "10000", "India"))

    df = pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "InvoiceDate",
            "Description",
            "Quantity",
            "UnitPrice",
            "CustomerID",
            "Country",
        ],
    )

    if include_noise and len(df) > 0:
        n_missing = max(1, int(0.003 * len(df)))
        missing_idx = rng.choice(len(df), size=min(n_missing, len(df)), replace=False)
        df.loc[df.index[missing_idx], "Description"] = np.nan

    return df
