"""Reusable helper functions for the Data Science Fundamentals Assessment.

The notebooks in ``assignments/`` import these functions (Assignment 10 shows
how) and the tests in ``tests/`` check them. Every function is small, has a
docstring and raises a clear error for invalid input.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "visualizations"
DEFAULT_SEED = 42


def load_csv(path, required_columns=None):
    """Load a CSV file into a DataFrame with basic validation.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file has no data rows or a required column is missing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file has no data rows: {path}")
    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    return df


def missing_report(df):
    """Return the number and percentage of missing values per column.

    Columns are sorted from the most to the least missing.
    """
    if df.empty:
        raise ValueError("Cannot build a missing-value report for an empty DataFrame")
    counts = df.isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": (counts / len(df) * 100).round(2)})
    return report.sort_values("missing", ascending=False)


def iqr_bounds(series, k=1.5):
    """Return (lower, upper) outlier fences using the IQR rule.

    A value is an outlier if it is below Q1 - k*IQR or above Q3 + k*IQR.
    """
    values = pd.Series(series).dropna()
    if values.empty:
        raise ValueError("Cannot compute IQR bounds of an empty series")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def iqr_outlier_mask(series, k=1.5):
    """Boolean mask that is True for IQR outliers (missing values are never outliers)."""
    series = pd.Series(series)
    lower, upper = iqr_bounds(series, k)
    return (series < lower) | (series > upper)


def clean_text(series):
    """Strip spaces, collapse repeated spaces and use Title Case for a text column."""
    return (
        pd.Series(series)
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def to_numeric_with_report(series):
    """Convert a column to numbers; unparseable entries become NaN.

    Returns (numeric_series, number_of_entries_that_could_not_be_converted).
    Entries that were already missing are not counted as failures.
    """
    series = pd.Series(series)
    numeric = pd.to_numeric(series, errors="coerce")
    failed = int((numeric.isna() & series.notna()).sum())
    return numeric, failed


def cohens_d(a, b):
    """Cohen's d effect size (pooled standard deviation) for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least two values")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var == 0:
        raise ValueError("Both groups have zero variance")
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def eta_squared(groups):
    """Share of the total variation explained by group membership (0 to 1).

    ``groups`` is a list of arrays, one per group. This is the effect size that
    belongs to a one-way ANOVA: SS_between / SS_total.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if len(arrays) < 2:
        raise ValueError("Need at least two groups")
    everything = np.concatenate(arrays)
    grand_mean = everything.mean()
    ss_total = ((everything - grand_mean) ** 2).sum()
    if ss_total == 0:
        raise ValueError("All values are identical")
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in arrays)
    return float(ss_between / ss_total)


def describe_numeric(df):
    """Summary table with mean, median, variance, standard deviation, quartiles and IQR."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("The DataFrame has no numeric columns")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    return pd.DataFrame(
        {
            "count": numeric.count(),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "variance": numeric.var(),
            "std": numeric.std(),
            "min": numeric.min(),
            "q1": q1,
            "q3": q3,
            "max": numeric.max(),
            "iqr": q3 - q1,
        }
    )


def save_figure(fig, filename, folder=FIGURE_DIR, dpi=150):
    """Save a Matplotlib figure as PNG (creating the folder if needed) and return the path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
