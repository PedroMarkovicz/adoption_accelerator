"""
Inference-specific visualization.

Prediction distribution and confidence histogram plots for
notebook 14 diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Palette consistent with training/interpretability notebooks
PALETTE = ["#4680a7", "#e07b54", "#5ea577", "#c75c8a", "#b5a642"]


def plot_prediction_distribution(
    predicted_classes: np.ndarray,
    training_distribution: dict[int, float] | None = None,
    save_path: str | Path | None = None,
    class_labels: dict[int, str] | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Side-by-side bar chart: test predictions vs. training distribution.

    Parameters
    ----------
    predicted_classes : ndarray
        Integer class predictions (0-4).
    training_distribution : dict or None
        Training class counts ``{class: count}``.
    save_path : str or Path or None
        If provided, save the figure as PNG.
    class_labels : dict or None
        Optional class label mapping.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    classes = list(range(5))
    n_test = len(predicted_classes)
    test_counts = np.bincount(predicted_classes, minlength=5)
    test_pct = test_counts / n_test * 100

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35
    x = np.arange(len(classes))

    if training_distribution is not None:
        train_total = sum(training_distribution.values())
        train_pct = np.array(
            [training_distribution.get(c, 0) / train_total * 100 for c in classes]
        )
        ax.bar(
            x - bar_width / 2,
            train_pct,
            bar_width,
            label="Training (actual)",
            color=PALETTE[0],
            alpha=0.85,
        )
        ax.bar(
            x + bar_width / 2,
            test_pct,
            bar_width,
            label="Test (predicted)",
            color=PALETTE[1],
            alpha=0.85,
        )

        for i in range(len(classes)):
            ax.text(
                x[i] - bar_width / 2,
                train_pct[i] + 0.5,
                f"{train_pct[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.text(
                x[i] + bar_width / 2,
                test_pct[i] + 0.5,
                f"{test_pct[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax.bar(
            x,
            test_pct,
            bar_width * 2,
            label="Test (predicted)",
            color=PALETTE[1],
            alpha=0.85,
        )
        for i in range(len(classes)):
            ax.text(
                x[i],
                test_pct[i] + 0.5,
                f"{test_pct[i]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Adoption Speed Class")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Prediction Distribution: Training vs. Test")
    x_labels = [
        f"{c}" if class_labels is None else f"{c}\n{class_labels.get(c, '')}"
        for c in classes
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Prediction distribution figure saved to %s", save_path)

    return fig


def plot_confidence_distribution(
    confidences: np.ndarray,
    save_path: str | Path | None = None,
    low_confidence_threshold: float = 0.30,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Histogram of prediction confidence values.

    Parameters
    ----------
    confidences : ndarray
        Maximum class probability for each sample.
    save_path : str or Path or None
        If provided, save the figure.
    low_confidence_threshold : float
        Threshold for shading low-confidence region.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        confidences,
        bins=50,
        color=PALETTE[0],
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    # Shade low-confidence region
    ax.axvspan(
        0,
        low_confidence_threshold,
        alpha=0.15,
        color="red",
        label=f"Low confidence (<{low_confidence_threshold})",
    )

    # Summary lines
    mean_conf = float(np.mean(confidences))
    median_conf = float(np.median(confidences))
    ax.axvline(
        mean_conf,
        color=PALETTE[1],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_conf:.3f}",
    )
    ax.axvline(
        median_conf,
        color=PALETTE[2],
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {median_conf:.3f}",
    )

    ax.set_xlabel("Confidence (max probability)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confidence distribution figure saved to %s", save_path)

    return fig
