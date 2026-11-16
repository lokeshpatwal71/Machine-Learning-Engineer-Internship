"""Automated tests for src/ds_helpers.py (run with: python -m pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import ds_helpers as h  # noqa: E402


@pytest.fixture
def messy():
    return pd.DataFrame(
        {
            "name": ["  alice  ", "BOB", "carol  smith", None],
            "score": ["90", "x", "75.5", None],
            "age": [20, 21, 22, 23],
        }
    )


# ---------------------------------------------------------------- load_csv
def test_load_csv_reads_file(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    df = h.load_csv(path, required_columns=["a", "b"])
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        h.load_csv(tmp_path / "nope.csv")


def test_load_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n")
    with pytest.raises(ValueError, match="no data rows"):
        h.load_csv(path)


def test_load_csv_missing_required_column(tmp_path):
    path = tmp_path / "cols.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        h.load_csv(path, required_columns=["a", "z"])


# ----------------------------------------------------------- missing_report
def test_missing_report_counts_and_sorts():
    df = pd.DataFrame({"x": [1, None, None, 4], "y": [1, 2, 3, None], "z": [1, 2, 3, 4]})
    rep = h.missing_report(df)
    assert list(rep.index) == ["x", "y", "z"]
    assert rep.loc["x", "missing"] == 2
    assert rep.loc["x", "percent"] == pytest.approx(50.0)
    assert rep.loc["z", "missing"] == 0


def test_missing_report_empty_frame():
    with pytest.raises(ValueError):
        h.missing_report(pd.DataFrame())


# --------------------------------------------------------------------- IQR
def test_iqr_bounds_known_values():
    lower, upper = h.iqr_bounds([1, 2, 3, 4, 5, 6, 7, 8])
    # Q1 = 2.75, Q3 = 6.25, IQR = 3.5 -> fences -2.5 and 11.5
    assert lower == pytest.approx(-2.5)
    assert upper == pytest.approx(11.5)


def test_iqr_outlier_mask_flags_extreme_value():
    values = pd.Series([10, 11, 12, 11, 10, 12, 11, 500])
    mask = h.iqr_outlier_mask(values)
    assert mask.tolist() == [False] * 7 + [True]


def test_iqr_outlier_mask_ignores_missing():
    values = pd.Series([10, 11, np.nan, 12, 11])
    assert not h.iqr_outlier_mask(values).any()


def test_iqr_bounds_empty_series():
    with pytest.raises(ValueError):
        h.iqr_bounds(pd.Series([], dtype=float))


# --------------------------------------------------------------- text/numeric
def test_clean_text_strips_and_titles(messy):
    out = h.clean_text(messy["name"])
    assert out.iloc[0] == "Alice"
    assert out.iloc[1] == "Bob"
    assert out.iloc[2] == "Carol Smith"
    assert pd.isna(out.iloc[3])


def test_to_numeric_with_report_counts_failures(messy):
    numeric, failed = h.to_numeric_with_report(messy["score"])
    assert failed == 1  # "x" failed; the None value was already missing
    assert numeric.iloc[0] == pytest.approx(90.0)
    assert numeric.iloc[2] == pytest.approx(75.5)
    assert numeric.isna().sum() == 2


# -------------------------------------------------------------- effect sizes
def test_cohens_d_known_value():
    a = [1, 2, 3, 4, 5]
    b = [3, 4, 5, 6, 7]
    # both variances are 2.5, means differ by -2 -> d = -2 / sqrt(2.5)
    assert h.cohens_d(a, b) == pytest.approx(-2 / np.sqrt(2.5))


def test_cohens_d_needs_two_values():
    with pytest.raises(ValueError):
        h.cohens_d([1], [2, 3])


def test_eta_squared_perfect_and_none():
    # Groups are completely separated -> all variation is between groups
    assert h.eta_squared([[1, 1, 1], [5, 5, 5]]) == pytest.approx(1.0)
    # Groups have identical means -> nothing is explained by the group
    assert h.eta_squared([[1, 2, 3], [1, 2, 3]]) == pytest.approx(0.0)


def test_eta_squared_validation():
    with pytest.raises(ValueError):
        h.eta_squared([[1, 2, 3]])
    with pytest.raises(ValueError):
        h.eta_squared([[2, 2], [2, 2]])


# ------------------------------------------------------------------ describe
def test_describe_numeric_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": list("abcde")})
    out = h.describe_numeric(df)
    assert list(out.index) == ["x"]
    assert out.loc["x", "mean"] == pytest.approx(3.0)
    assert out.loc["x", "median"] == pytest.approx(3.0)
    assert out.loc["x", "variance"] == pytest.approx(2.5)
    assert out.loc["x", "iqr"] == pytest.approx(2.0)


def test_describe_numeric_requires_numbers():
    with pytest.raises(ValueError):
        h.describe_numeric(pd.DataFrame({"a": ["x", "y"]}))


# -------------------------------------------------------------- save_figure
def test_save_figure_creates_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    path = h.save_figure(fig, "chart.png", tmp_path / "figs")
    plt.close(fig)
    assert path.is_file()
    assert path.stat().st_size > 1000
