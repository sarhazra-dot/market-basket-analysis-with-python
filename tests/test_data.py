import numpy as np
import pandas as pd
import pytest

from marketbasket.data import (
    NOISE_ITEMS,
    PATTERNS,
    PRICES,
    clean_transactions,
    generate_synthetic_transactions,
    load_transactions,
)


# ------------------------------ synthetic data ------------------------------ #
def test_generator_is_deterministic():
    a = generate_synthetic_transactions(300, seed=1)
    b = generate_synthetic_transactions(300, seed=1)
    c = generate_synthetic_transactions(300, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_generator_schema_and_size():
    df = generate_synthetic_transactions(200, seed=3)
    assert list(df.columns) == [
        "InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "CustomerID", "Country",
    ]
    regular = df[~df["InvoiceNo"].str.startswith("C")]
    assert regular["InvoiceNo"].nunique() == 200


def test_generator_no_noise_is_clean():
    df = generate_synthetic_transactions(200, seed=3, include_noise=False)
    assert df["Description"].notna().all()
    assert (df["Quantity"] > 0).all()
    assert "POSTAGE" not in set(df["Description"])
    assert not df["InvoiceNo"].str.startswith("C").any()


def test_generator_noise_contains_dirty_rows():
    df = generate_synthetic_transactions(1000, seed=3, include_noise=True)
    assert (df["Quantity"] < 0).any()
    assert df["InvoiceNo"].str.startswith("C").any()
    assert df["Description"].isna().any()
    assert "POSTAGE" in set(df["Description"])


def test_generator_rejects_bad_input():
    with pytest.raises(ValueError):
        generate_synthetic_transactions(0)


def test_catalog_is_consistent():
    pattern_items = [item for _, _, items in PATTERNS for item, _ in items]
    # No item may appear in two missions or in both missions and noise.
    assert len(pattern_items) == len(set(pattern_items))
    assert not set(pattern_items) & set(NOISE_ITEMS)
    assert len(NOISE_ITEMS) == len(set(NOISE_ITEMS))
    for item in pattern_items + list(NOISE_ITEMS):
        assert item in PRICES
    for _, _, items in PATTERNS:
        assert all(0 < p <= 1 for _, p in items)


# --------------------------------- cleaning --------------------------------- #
def _frame(rows):
    return pd.DataFrame(rows, columns=["invoice", "item", "quantity", "price"])


def test_clean_removes_invalid_rows():
    df = _frame(
        [
            ("1", "A", 1, 1.0),
            ("1", "B", 1, 1.0),
            ("C2", "A", -1, 1.0),   # cancellation
            ("C2", "B", -1, 1.0),
            ("3", "A", -2, 1.0),    # negative quantity
            ("3", "B", 1, 1.0),
            ("3", "C", 1, 0.0),     # zero price
            ("3", "POSTAGE", 1, 5.0),  # not a product
            ("3", "D", 1, 2.0),
        ]
    )
    out = clean_transactions(df)
    # Invoice C2 is a cancellation. In invoice 3 the negative-quantity A, zero-price C
    # and POSTAGE rows are dropped, leaving the valid items B and D.
    assert set(zip(out["invoice"], out["item"])) == {("1", "A"), ("1", "B"), ("3", "B"), ("3", "D")}


def test_clean_handles_missing_values_and_whitespace():
    df = pd.DataFrame(
        {
            "invoice": ["1", "1", None, "1", "2", "2"],
            "item": ["  Red   Mug ", "Blue Mug", "X", None, "", "Blue Mug"],
        }
    )
    out = clean_transactions(df, min_basket_size=1)
    assert set(zip(out["invoice"], out["item"])) == {("1", "Red Mug"), ("1", "Blue Mug"), ("2", "Blue Mug")}


def test_clean_deduplicates_within_basket():
    df = pd.DataFrame({"invoice": ["1", "1", "1", "1"], "item": ["A", "A", "B", "B"]})
    out = clean_transactions(df)
    assert len(out) == 2


def test_clean_min_basket_size():
    df = pd.DataFrame({"invoice": ["1", "2", "2", "3", "3", "3"], "item": ["A", "A", "B", "A", "B", "C"]})
    assert set(clean_transactions(df, min_basket_size=2)["invoice"]) == {"2", "3"}
    assert set(clean_transactions(df, min_basket_size=3)["invoice"]) == {"3"}
    assert set(clean_transactions(df, min_basket_size=1)["invoice"]) == {"1", "2", "3"}


def test_clean_optional_flags():
    df = pd.DataFrame(
        {"invoice": ["C1", "C1", "2", "2"], "item": ["A", "B", "POSTAGE", "A"]}
    )
    kept = clean_transactions(df, min_basket_size=1, drop_cancellations=False, drop_non_products=False)
    assert len(kept) == 4


def test_clean_keeps_non_cancellation_c_ids():
    df = pd.DataFrame({"invoice": ["CUST-1", "CUST-1"], "item": ["A", "B"]})
    assert len(clean_transactions(df)) == 2


def test_clean_validates_input():
    with pytest.raises(ValueError):
        clean_transactions(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        clean_transactions(pd.DataFrame({"invoice": ["1"], "item": ["A"]}), min_basket_size=0)


def test_clean_returns_two_columns_and_clean_index():
    out = clean_transactions(_frame([("1", "A", 1, 1.0), ("1", "B", 1, 1.0)]))
    assert list(out.columns) == ["invoice", "item"]
    assert list(out.index) == [0, 1]


# ---------------------------------- loading --------------------------------- #
def test_load_autodetects_online_retail_columns(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("InvoiceNo,StockCode,Description,Quantity,UnitPrice\n536365,1,MUG,6,2.5\n536365,2,PEN,1,0.5\n")
    df = load_transactions(path)
    assert list(df.columns) == ["invoice", "item", "quantity", "price"]
    assert df["invoice"].tolist() == ["536365", "536365"]
    assert df["quantity"].tolist() == [6, 1]


def test_load_autodetects_online_retail_ii_columns(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("Invoice,StockCode,Description,Quantity,Price,Customer ID\n1,1,MUG,6,2.5,1\n")
    df = load_transactions(path)
    assert list(df.columns) == ["invoice", "item", "quantity", "price"]


def test_load_explicit_columns_and_optional_columns_missing(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("basket,sku\n1,a\n1,b\n")
    df = load_transactions(path, invoice_col="basket", item_col="sku")
    assert list(df.columns) == ["invoice", "item"]


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "missing.csv")
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="invoice"):
        load_transactions(path)
    with pytest.raises(ValueError, match="not found"):
        load_transactions(path, invoice_col="nope", item_col="b")


def test_load_latin1_file(tmp_path):
    path = tmp_path / "t.csv"
    path.write_bytes("InvoiceNo,Description\n1,CAF\u00c9 MUG\n1,PEN\n".encode("latin-1"))
    df = load_transactions(path)
    assert "CAF\u00c9 MUG" in df["item"].tolist()


def test_load_invalid_numbers_become_nan(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("InvoiceNo,Description,Quantity,UnitPrice\n1,A,x,1\n1,B,2,1\n")
    df = load_transactions(path)
    assert np.isnan(df["quantity"].iloc[0])
    cleaned = clean_transactions(df, min_basket_size=1)
    assert cleaned["item"].tolist() == ["B"]


def test_full_synthetic_roundtrip_through_csv(tmp_path):
    raw = generate_synthetic_transactions(300, seed=5)
    path = tmp_path / "s.csv"
    raw.to_csv(path, index=False)
    cleaned = clean_transactions(load_transactions(path))
    assert not cleaned["invoice"].str.startswith("C").any()
    assert "POSTAGE" not in set(cleaned["item"])
    assert cleaned["item"].notna().all()


def test_load_excel_reads_all_sheets(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "t.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {"Invoice": [1, 1], "Description": ["A", "B"], "Quantity": [1, 1], "Price": [1.0, 2.0]}
        ).to_excel(writer, sheet_name="Year 1", index=False)
        pd.DataFrame(
            {"Invoice": [2, 2], "Description": ["A", "C"], "Quantity": [1, 1], "Price": [1.0, 2.0]}
        ).to_excel(writer, sheet_name="Year 2", index=False)
    df = load_transactions(path)
    assert df["invoice"].tolist() == ["1", "1", "2", "2"]
    assert df["item"].tolist() == ["A", "B", "A", "C"]
    assert df["price"].tolist() == [1.0, 2.0, 1.0, 2.0]
    assert len(clean_transactions(df)) == 4
