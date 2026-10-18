import pandas as pd
import pytest

from marketbasket.data import clean_transactions, generate_synthetic_transactions
from marketbasket.preprocess import build_baskets, encode_baskets


@pytest.fixture(scope="session")
def raw_synthetic():
    return generate_synthetic_transactions(n_transactions=2000, seed=7)


@pytest.fixture(scope="session")
def synthetic_baskets(raw_synthetic):
    standardized = pd.DataFrame(
        {
            "invoice": raw_synthetic["InvoiceNo"],
            "item": raw_synthetic["Description"],
            "quantity": raw_synthetic["Quantity"],
            "price": raw_synthetic["UnitPrice"],
        }
    )
    clean = clean_transactions(standardized)
    return build_baskets(clean)


@pytest.fixture(scope="session")
def synthetic_onehot(synthetic_baskets):
    return encode_baskets(synthetic_baskets, min_item_support=0.01)


@pytest.fixture()
def toy_baskets():
    """Five baskets with hand-computable statistics (n = 5)."""
    return [
        ["bread", "butter"],
        ["bread", "butter", "milk"],
        ["bread", "milk"],
        ["butter"],
        ["eggs", "milk"],
    ]
