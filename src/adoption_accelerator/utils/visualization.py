"""
Visualization and statistical summary utilities for the Adoption Accelerator project.

Shared across EDA notebooks (04, 05, 06) and the Streamlit frontend.
All plotting functions return ``matplotlib.figure.Figure`` objects for downstream
persistence via :func:`save_figure`.

Functions
---------
plot_target_distribution(y)
    Plot AdoptionSpeed class distribution.
plot_numeric_distribution(df, col, target_col)
    Histogram + box plot for a numeric column, optionally grouped by target.
plot_categorical_distribution(df, col, target_col)
    Bar chart for a categorical column, optionally grouped by target.
plot_correlation_matrix(df, columns, method)
    Correlation heatmap.
compute_cramers_v(df, col1, col2)
    Cramér's V statistic for two categorical columns.
compute_descriptive_stats(df, columns)
    Descriptive statistics (mean, median, std, skew, kurtosis).
save_figure(fig, name, subdir)
    Save a figure to ``reports/figures/{subdir}/{name}.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from adoption_accelerator import config as cfg

logger = logging.getLogger("adoption_accelerator")

# ── Visual defaults ─────────────────────────────────────────────────

ADOPTION_SPEED_LABELS = {
    0: "Same day",
    1: "1–7 days",
    2: "8–30 days",
    3: "31–90 days",
    4: "100+ days",
}

ADOPTION_SPEED_PALETTE = {
    0: "#4dec8f",
    1: "#359760",
    2: "#f39c12",
    3: "#e67e22",
    4: "#e74c3c",
}

TARGET_COL_NAME = "AdoptionSpeed"


# ── Public API ──────────────────────────────────────────────────────


def plot_target_distribution(
    y: pd.Series,
    *,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot and return the AdoptionSpeed class distribution chart.

    Parameters
    ----------
    y : pd.Series
        Series of AdoptionSpeed values (0–4).
    figsize : tuple, optional
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = y.value_counts().sort_index()
    pcts = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [ADOPTION_SPEED_PALETTE.get(i, "#95a5a6") for i in counts.index]
    bars = ax.bar(
        counts.index, counts.values, color=colors, edgecolor="white", width=0.7
    )

    for bar, pct in zip(bars, pcts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{int(bar.get_height()):,}\n({pct}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Adoption Speed Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Target Variable Distribution — AdoptionSpeed", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(counts.index)
    ax.set_xticklabels(
        [f"{i}\n{ADOPTION_SPEED_LABELS.get(i, '')}" for i in counts.index],
        fontsize=10,
    )
    ax.set_ylim(0, counts.max() * 1.18)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_numeric_distribution(
    df: pd.DataFrame,
    col: str,
    target_col: Optional[str] = None,
    *,
    figsize: tuple[float, float] = (14, 5),
    bins: int = 18,
) -> plt.Figure:
    """
    Histogram + box plot for a numeric column.

    Uses side-by-side (dodged) histograms when grouped by target
    to avoid overlapping classes.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ─────────────────────────────────────────────
    # 1️⃣ Histogram (side-by-side bars)
    # ─────────────────────────────────────────────
    if target_col:
        classes = sorted(df[target_col].dropna().unique())
        n_classes = len(classes)

        data = df[[col, target_col]].dropna()

        # Compute global bin edges
        counts, bin_edges = np.histogram(data[col], bins=bins)

        bin_width = bin_edges[1] - bin_edges[0]
        class_width = bin_width / n_classes

        for i, cls in enumerate(classes):
            subset = data.loc[data[target_col] == cls, col]

            hist_counts, _ = np.histogram(subset, bins=bin_edges)

            # Shift each class inside the bin
            shift = (i - n_classes / 2) * class_width + class_width / 2

            axes[0].bar(
                bin_edges[:-1] + shift,
                hist_counts,
                width=class_width,
                color=ADOPTION_SPEED_PALETTE.get(cls),
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                label=f"Class {cls}",
                align="edge",
            )

        axes[0].legend(
            title=target_col,
            fontsize=9,
            title_fontsize=10,
            frameon=False,
        )

    else:
        axes[0].hist(
            df[col].dropna(),
            bins=bins,
            color="#3498db",
            edgecolor="white",
            linewidth=0.5,
        )

    axes[0].set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(col, fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)

    # ─────────────────────────────────────────────
    # 2️⃣ Box Plot
    # ─────────────────────────────────────────────
    if target_col:
        palette = [
            ADOPTION_SPEED_PALETTE.get(c, "#95a5a6")
            for c in sorted(df[target_col].unique())
        ]

        sns.boxplot(
            data=df,
            x=target_col,
            y=col,
            palette=palette,
            ax=axes[1],
            fliersize=2,
            linewidth=1.2,
        )

        axes[1].set_title(f"{col} by {target_col}", fontsize=13, fontweight="bold")
    else:
        sns.boxplot(
            data=df,
            y=col,
            color="#3498db",
            ax=axes[1],
            fliersize=2,
        )
        axes[1].set_title(f"Box Plot — {col}", fontsize=13, fontweight="bold")

    sns.despine(ax=axes[0])
    sns.despine(ax=axes[1])

    fig.tight_layout()
    return fig


def plot_categorical_distribution(
    df: pd.DataFrame,
    col: str,
    target_col: Optional[str] = None,
    *,
    figsize: tuple[float, float] = (10, 5),
    top_n: Optional[int] = None,
    label_map: Optional[dict] = None,
) -> plt.Figure:
    """Generate bar chart for a categorical column.

    If *target_col* is provided, produces a stacked proportional bar chart
    with percentage labels inside each segment.
    """

    data = df.copy()

    if top_n is not None:
        top_cats = data[col].value_counts().head(top_n).index
        data = data[data[col].isin(top_cats)]

    fig, ax = plt.subplots(figsize=figsize)

    if target_col:
        # ----- Stacked proportional bar -----
        ct = pd.crosstab(data[col], data[target_col], normalize="index") * 100
        ct = ct.sort_index()

        colors = [ADOPTION_SPEED_PALETTE.get(c, "#95a5a6") for c in ct.columns]

        ct.plot(
            kind="bar",
            stacked=True,
            color=colors,
            ax=ax,
            edgecolor="white",
            width=0.75,
        )

        ax.set_ylabel("Proportion (%)", fontsize=11)

        # Add percentage labels inside bars
        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                if height > 5:  # avoid clutter for very small segments
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        f"{height:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                        fontweight="bold",
                    )

        ax.legend(
            title=target_col,
            fontsize=9,
            title_fontsize=10,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    else:
        # ----- Standard count bar -----
        counts = data[col].value_counts().sort_index()

        bars = ax.bar(
            range(len(counts)),
            counts.values,
            color="#3498db",
            edgecolor="white",
        )

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, fontsize=10)
        ax.set_ylabel("Count", fontsize=11)

        # Add count labels on top
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # ----- Label mapping -----
    if label_map:
        current_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        new_labels = []
        for lbl in current_labels:
            try:
                key = int(lbl) if lbl.lstrip("-").isdigit() else lbl
            except (ValueError, AttributeError):
                key = lbl
            new_labels.append(str(label_map.get(key, lbl)))
        ax.set_xticklabels(new_labels, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel(col, fontsize=11)
    ax.set_title(
        f"{'Distribution' if not target_col else target_col + ' by'} {col}",
        fontsize=13,
        fontweight="bold",
    )

    sns.despine(ax=ax)
    fig.tight_layout()

    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "pearson",
    *,
    figsize: tuple[float, float] = (12, 10),
) -> plt.Figure:
    """Compute and visualize a correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str, optional
        Subset of columns. If None, uses all numeric columns.
    method : str
        Correlation method (``"pearson"``, ``"spearman"``, ``"kendall"``).
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if columns is not None:
        data = df[columns]
    else:
        data = df.select_dtypes(include="number")

    corr = data.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontsize": 8},
    )
    ax.set_title(
        f"Correlation Matrix ({method.title()})", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    return fig


def compute_cramers_v(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Compute Cramér's V statistic for two categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    col1, col2 : str
        Column names.

    Returns
    -------
    float
        Cramér's V ∈ [0, 1].
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def compute_cramers_v_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Compute a pairwise Cramér's V matrix for several categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : sequence of str

    Returns
    -------
    pd.DataFrame
        Symmetric matrix of Cramér's V values.
    """
    n = len(columns)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = compute_cramers_v(df, columns[i], columns[j])
            matrix[i, j] = v
            matrix[j, i] = v
    return pd.DataFrame(matrix, index=columns, columns=columns)


def compute_descriptive_stats(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a DataFrame of descriptive statistics for specified numeric columns.

    Includes: count, mean, median, std, min, max, skewness, kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
    columns : sequence of str

    Returns
    -------
    pd.DataFrame
        Rows = statistics, Columns = features.
    """
    result: dict[str, dict[str, float]] = {}
    for col in columns:
        s = df[col].dropna()
        result[col] = {
            "count": len(s),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skewness": float(s.skew()),
            "kurtosis": float(s.kurtosis()),
        }
    return pd.DataFrame(result).T


def save_figure(
    fig: plt.Figure,
    name: str,
    subdir: str = "eda_tabular",
    *,
    dpi: int = 150,
) -> Path:
    """Save a matplotlib/seaborn figure to ``reports/figures/{subdir}/{name}.png``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Filename without extension.
    subdir : str
        Subdirectory under ``reports/figures/``.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to the saved file.
    """
    out_dir = cfg.REPORTS_FIGURES / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info("Saved figure: %s", path)
    return path


# ── Text & sentiment visualizations ─────────────────────────────────


def compute_text_statistics(descriptions: pd.Series) -> pd.DataFrame:
    """Compute character length, word count, and sentence count for a Series of text.

    Parameters
    ----------
    descriptions : pd.Series
        Series of text strings (may contain NaN).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``char_length``, ``word_count``, ``sentence_count``.
        Index matches the input Series index.
    """
    import re

    filled = descriptions.fillna("")
    char_length = filled.str.len()
    word_count = filled.str.split().str.len().fillna(0).astype(int)
    sentence_count = filled.apply(
        lambda t: len(re.split(r"[.!?]+", t.strip())) - 1 if t.strip() else 0
    ).clip(lower=0)

    return pd.DataFrame(
        {
            "char_length": char_length,
            "word_count": word_count,
            "sentence_count": sentence_count,
        },
        index=descriptions.index,
    )


def plot_text_length_distributions(
    stats_df: pd.DataFrame,
    target: pd.Series | None = None,
    *,
    figsize: tuple[float, float] = (18, 10),
    bins: int = 15,
) -> plt.Figure:
    """
    Plot text length distributions with clearly separated AdoptionSpeed classes.

    Uses side-by-side (dodged) histograms when grouped by target
    to avoid overlapping classes.
    """

    cols = ["char_length", "word_count", "sentence_count"]
    titles = ["Character Length", "Word Count", "Sentence Count"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    for i, (col, title) in enumerate(zip(cols, titles)):
        # ─────────────────────────────────────────────
        # 1️⃣ Histogram (side-by-side bars)
        # ─────────────────────────────────────────────
        ax_hist = axes[0, i]

        if target is not None:
            data = pd.DataFrame({col: stats_df[col], TARGET_COL_NAME: target}).dropna()
            classes = sorted(data[TARGET_COL_NAME].unique())
            n_classes = len(classes)

            # Global bins
            _, bin_edges = np.histogram(data[col], bins=bins)
            bin_width = bin_edges[1] - bin_edges[0]
            class_width = (bin_width / n_classes) * 0.9

            for j, cls in enumerate(classes):
                subset = data.loc[data[TARGET_COL_NAME] == cls, col]
                hist_counts, _ = np.histogram(subset, bins=bin_edges)

                shift = (j - n_classes / 2) * class_width + class_width / 2

                ax_hist.bar(
                    bin_edges[:-1] + shift,
                    hist_counts,
                    width=class_width,
                    color=ADOPTION_SPEED_PALETTE.get(cls),
                    edgecolor="white",
                    linewidth=0.5,
                    label=f"Class {cls}",
                    align="edge",
                )

            ax_hist.legend(
                title=TARGET_COL_NAME,
                fontsize=8,
                title_fontsize=9,
                frameon=False,
            )

        else:
            ax_hist.hist(
                stats_df[col].dropna(),
                bins=bins,
                color="#3498db",
                edgecolor="white",
                linewidth=0.5,
            )

        ax_hist.set_title(f"Distribution of {title}", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel(title, fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        sns.despine(ax=ax_hist)

        # ─────────────────────────────────────────────
        # 2️⃣ Box Plot
        # ─────────────────────────────────────────────
        ax_box = axes[1, i]

        if target is not None:
            plot_df = pd.DataFrame({col: stats_df[col], TARGET_COL_NAME: target})

            palette = [
                ADOPTION_SPEED_PALETTE.get(c, "#95a5a6")
                for c in sorted(target.unique())
            ]

            sns.boxplot(
                data=plot_df,
                x=TARGET_COL_NAME,
                y=col,
                palette=palette,
                ax=ax_box,
                fliersize=2,
                linewidth=1.2,
            )

            ax_box.set_title(
                f"{title} by {TARGET_COL_NAME}",
                fontsize=11,
                fontweight="bold",
            )

        else:
            sns.boxplot(
                data=stats_df,
                y=col,
                color="#3498db",
                ax=ax_box,
                fliersize=2,
            )
            ax_box.set_title(f"Box Plot — {title}", fontsize=11, fontweight="bold")

        sns.despine(ax=ax_box)

    fig.suptitle("Text Length Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    return fig


def plot_sentiment_distributions(
    sentiment_df: pd.DataFrame,
    target: pd.Series | None = None,
    *,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot sentiment score and magnitude distributions grouped by target.

    Creates a 2×2 grid: histograms (top) and box plots (bottom) for
    ``doc_sentiment_score`` and ``doc_sentiment_magnitude``.

    Parameters
    ----------
    sentiment_df : pd.DataFrame
        DataFrame with ``doc_sentiment_score`` and ``doc_sentiment_magnitude``.
    target : pd.Series, optional
        AdoptionSpeed values for grouping.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = ["doc_sentiment_score", "doc_sentiment_magnitude"]
    titles = ["Document Sentiment Score", "Document Sentiment Magnitude"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for i, (col, title) in enumerate(zip(cols, titles)):
        ax_hist = axes[0, i]
        if target is not None:
            for cls in sorted(target.unique()):
                mask = target == cls
                ax_hist.hist(
                    sentiment_df.loc[mask, col].dropna(),
                    bins=40,
                    alpha=0.5,
                    label=f"Class {cls}",
                    color=ADOPTION_SPEED_PALETTE.get(cls, None),
                )
            ax_hist.legend(fontsize=8)
        else:
            ax_hist.hist(
                sentiment_df[col].dropna(), bins=40, color="#3498db", edgecolor="white"
            )
        ax_hist.set_title(f"Distribution of {title}", fontsize=11, fontweight="bold")
        ax_hist.set_xlabel(title, fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        sns.despine(ax=ax_hist)

        ax_box = axes[1, i]
        if target is not None:
            plot_df = pd.DataFrame({col: sentiment_df[col], TARGET_COL_NAME: target})
            palette = [
                ADOPTION_SPEED_PALETTE.get(c, "#95a5a6")
                for c in sorted(target.unique())
            ]
            sns.boxplot(
                data=plot_df,
                x=TARGET_COL_NAME,
                y=col,
                palette=palette,
                ax=ax_box,
                fliersize=2,
            )
            ax_box.set_title(
                f"{title} by {TARGET_COL_NAME}", fontsize=11, fontweight="bold"
            )
        else:
            sns.boxplot(
                data=sentiment_df, y=col, color="#3498db", ax=ax_box, fliersize=2
            )
            ax_box.set_title(f"Box Plot — {title}", fontsize=11, fontweight="bold")
        sns.despine(ax=ax_box)

    fig.suptitle("Sentiment Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ── Image & metadata visualizations ────────────────────────────────


def plot_image_grid(
    images: Sequence,
    titles: Sequence[str] | None = None,
    nrows: int = 2,
    ncols: int = 4,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Display a grid of images with optional titles.

    Parameters
    ----------
    images : sequence
        PIL Images, NumPy arrays, or file paths.
    titles : sequence of str, optional
        Per-image titles. If ``None``, images are shown without titles.
    nrows, ncols : int
        Grid dimensions.
    figsize : tuple, optional
        Matplotlib figure size (defaults to ``(3*ncols, 3*nrows)``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from PIL import Image as PILImage  # local import to keep module light

    if figsize is None:
        figsize = (3 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.asarray(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx < len(images):
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = PILImage.open(img)
            ax.imshow(np.asarray(img))
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_label_frequency(
    label_counts: pd.Series,
    top_n: int = 20,
    *,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Plot a horizontal bar chart of the top-N Vision API label frequencies.

    Parameters
    ----------
    label_counts : pd.Series
        Index = label text, values = counts.
    top_n : int
        Number of labels to display.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    top = label_counts.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(top)), top.values, color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=10)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_title(
        f"Top {top_n} Vision API Labels",
        fontsize=14,
        fontweight="bold",
    )

    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + top.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}",
            va="center",
            fontsize=9,
        )

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
