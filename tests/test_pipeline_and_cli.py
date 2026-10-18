from pathlib import Path

import pandas as pd
import pytest

from marketbasket import PipelineConfig, __version__, run_pipeline
from marketbasket.cli import main
from marketbasket.data import generate_synthetic_transactions
from marketbasket.rules import load_rules

EXPECTED_FILES = {
    "frequent_itemsets.csv",
    "rules.csv",
    "summary.md",
    "top_items.png",
    "support_confidence.png",
    "top_rules.png",
    "lift_heatmap.png",
}


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    generate_synthetic_transactions(1500, seed=11).to_csv(path, index=False)
    return path


def test_pipeline_writes_all_outputs(tmp_path, sample_csv):
    result = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "out"))
    assert {p.name for p in result.files} == EXPECTED_FILES
    assert all(p.exists() and p.stat().st_size > 0 for p in result.files)

    s = result.stats
    assert s["n_rows_clean"] < s["n_rows_raw"]  # cancellations / postage / missing removed
    assert 0 < s["n_baskets"] <= 1500
    assert s["n_rules"] == len(result.rules) > 0
    assert s["n_itemsets"] == len(result.itemsets)

    summary = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "# Market Basket Analysis Summary" in summary and "| Rule |" in summary


def test_pipeline_is_reproducible(tmp_path, sample_csv):
    a = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "a"))
    b = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "b"))
    assert (tmp_path / "a" / "rules.csv").read_bytes() == (tmp_path / "b" / "rules.csv").read_bytes()
    assert a.stats == b.stats


def test_pipeline_apriori_matches_fpgrowth(tmp_path, sample_csv):
    fp = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "f", algorithm="fpgrowth", min_support=0.02))
    ap = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "a", algorithm="apriori", min_support=0.02))
    assert fp.stats["n_itemsets"] == ap.stats["n_itemsets"]
    assert fp.stats["n_rules"] == ap.stats["n_rules"]


def test_pipeline_handles_no_rules(tmp_path, sample_csv):
    result = run_pipeline(PipelineConfig(input_path=sample_csv, output_dir=tmp_path / "none", min_support=0.95))
    assert result.rules.empty
    names = {p.name for p in result.files}
    assert {"rules.csv", "summary.md", "frequent_itemsets.csv"} <= names
    assert "top_rules.png" not in names
    assert "No rules met the thresholds" in (tmp_path / "none" / "summary.md").read_text(encoding="utf-8")


def test_pipeline_errors_when_nothing_survives_cleaning(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("InvoiceNo,Description\n1,A\n2,B\n")  # single-item baskets only
    with pytest.raises(ValueError, match="No transactions left"):
        run_pipeline(PipelineConfig(input_path=path, output_dir=tmp_path / "o"))


def test_pipeline_with_custom_columns(tmp_path):
    path = tmp_path / "custom.csv"
    rows = ["order,sku"] + [f"{i},{item}" for i in range(40) for item in ("tea", "milk")] + ["99,tea"]
    path.write_text("\n".join(rows) + "\n")
    result = run_pipeline(
        PipelineConfig(input_path=path, output_dir=tmp_path / "o", invoice_col="order", item_col="sku", min_support=0.1)
    )
    assert result.stats["n_baskets"] == 40
    assert len(result.rules) == 2


# ----------------------------------- CLI ------------------------------------ #
def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "d" / "tx.csv"
    assert main(["generate-data", "--out", str(data), "--n-transactions", "800", "--seed", "3"]) == 0
    assert data.exists()
    assert "Wrote" in capsys.readouterr().out

    out = tmp_path / "out"
    assert main(["run", "--input", str(data), "--output-dir", str(out), "--min-support", "0.02"]) == 0
    printed = capsys.readouterr().out
    assert "Top 5 rules" in printed and "lift=" in printed
    assert {p.name for p in out.iterdir()} == EXPECTED_FILES

    assert main(["recommend", "--rules", str(out / "rules.csv"), "--item", "spaghetti"]) == 0
    printed = capsys.readouterr().out
    assert "Tomato Sauce" in printed and "Recommended additions" in printed


def test_cli_recommend_warns_about_unknown_items(tmp_path, capsys, sample_csv):
    out = tmp_path / "out"
    main(["run", "--input", str(sample_csv), "--output-dir", str(out)])
    capsys.readouterr()
    assert main(["recommend", "--rules", str(out / "rules.csv"), "--item", "Unicorn Food"]) == 0
    captured = capsys.readouterr()
    assert "Unicorn Food" in captured.err
    assert "No recommendations" in captured.out


def test_cli_reports_errors_cleanly(tmp_path, capsys):
    assert main(["run", "--input", str(tmp_path / "missing.csv")]) == 1
    assert "Error:" in capsys.readouterr().err
    assert main(["recommend", "--rules", str(tmp_path / "missing.csv"), "--item", "x"]) == 1
    assert "Error:" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert main(["run", "--input", str(bad)]) == 1
    assert "Could not detect" in capsys.readouterr().err
    assert main(["run", "--input", str(bad), "--min-support", "5"]) == 1


def test_cli_version_and_required_args(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_committed_sample_data_is_usable(tmp_path):
    """The dataset shipped in data/ must load and produce the headline rules.

    (Not compared byte-for-byte with the generator: NumPy does not guarantee
    identical random streams across versions.)
    """
    committed = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"
    assert committed.exists()
    raw = pd.read_csv(committed, dtype=str)
    assert list(raw.columns) == [
        "InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "CustomerID", "Country",
    ]
    assert raw["InvoiceNo"].nunique() > 3000  # 3000 sales + cancellation invoices

    result = run_pipeline(PipelineConfig(input_path=committed, output_dir=tmp_path / "out"))
    assert result.stats["n_baskets"] > 2800
    pairs = {(frozenset(a), frozenset(c)) for a, c in zip(result.rules["antecedents"], result.rules["consequents"])}
    assert (frozenset({"Spaghetti"}), frozenset({"Tomato Sauce"})) in pairs
    assert (frozenset({"Diapers"}), frozenset({"Baby Wipes"})) in pairs


def test_quickstart_example_runs(capsys):
    import runpy

    example = Path(__file__).resolve().parent.parent / "examples" / "quickstart.py"
    runpy.run_path(str(example), run_name="__main__")
    out = capsys.readouterr().out
    assert "Top rules:" in out and "Tomato Sauce" in out
