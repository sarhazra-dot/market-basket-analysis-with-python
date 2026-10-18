import pandas as pd
import pytest

from marketbasket.preprocess import build_baskets, encode_baskets


def test_build_baskets_groups_sorts_and_dedupes():
    df = pd.DataFrame(
        {"invoice": ["2", "1", "1", "2", "2"], "item": ["b", "z", "a", "b", "c"]}
    )
    assert build_baskets(df) == [["a", "z"], ["b", "c"]]


def test_build_baskets_requires_columns():
    with pytest.raises(ValueError):
        build_baskets(pd.DataFrame({"x": [1]}))


def test_encode_shape_and_values(toy_baskets):
    onehot = encode_baskets(toy_baskets)
    assert onehot.shape == (5, 4)
    assert onehot.dtypes.apply(lambda d: d == bool).all()
    assert onehot["bread"].sum() == 3
    assert onehot["eggs"].sum() == 1


def test_min_item_support_filters_columns_but_keeps_all_baskets(toy_baskets):
    onehot = encode_baskets(toy_baskets, min_item_support=0.5)
    assert set(onehot.columns) == {"bread", "butter", "milk"}  # eggs only 20%
    assert len(onehot) == 5  # denominators are preserved


def test_max_items_keeps_most_frequent(toy_baskets):
    onehot = encode_baskets(toy_baskets, max_items=2)
    assert len(onehot.columns) == 2
    assert "eggs" not in onehot.columns


def test_encode_validation(toy_baskets):
    with pytest.raises(ValueError):
        encode_baskets([])
    with pytest.raises(ValueError):
        encode_baskets(toy_baskets, min_item_support=1.5)
    with pytest.raises(ValueError):
        encode_baskets(toy_baskets, max_items=0)


def test_everything_filtered_gives_zero_columns(toy_baskets):
    onehot = encode_baskets(toy_baskets, min_item_support=0.99)
    assert onehot.shape == (5, 0)
