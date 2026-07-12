"""Reliability and confidence diagnostics for the sentiment classifier.

A trained classifier that reports 0.95 confidence should be right about 95 percent
of the time. These helpers measure how far a model departs from that ideal using a
binned reliability curve and the expected calibration error, and render the
project's hero figure: a reliability diagram paired with a confidence histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability statistics for predicted-class confidence.

    Attributes:
        bin_edges: The ``n_bins + 1`` bin boundaries over ``[0, 1]``.
        bin_confidence: Mean predicted confidence within each bin (NaN if empty).
        bin_accuracy: Empirical accuracy within each bin (NaN if empty).
        bin_count: Number of samples in each bin.
        ece: Expected calibration error, the count-weighted mean gap between
            confidence and accuracy across non-empty bins.
    """

    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    ece: float


def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> ReliabilityCurve:
    """Compute a binned reliability curve and the expected calibration error.

    Args:
        probabilities: Per-class softmax probabilities, shape ``(n, num_classes)``.
        labels: True class indices, shape ``(n,)``.
        n_bins: Number of equal-width confidence bins over ``[0, 1]``.

    Returns:
        A :class:`ReliabilityCurve`.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predictions == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    ece = 0.0
    total = len(confidence)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence <= hi)
        n_in = int(in_bin.sum())
        count[b] = n_in
        if n_in == 0:
            continue
        conf[b] = float(confidence[in_bin].mean())
        acc[b] = float(correct[in_bin].mean())
        ece += (n_in / total) * abs(conf[b] - acc[b])

    return ReliabilityCurve(edges, conf, acc, count, float(ece))


def plot_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
    labels_names: tuple[str, ...] = ("negative", "positive"),
    title: str = "LSTM sentiment reliability",
) -> ReliabilityCurve:
    """Render the reliability-diagram hero figure and return the curve.

    The figure has three panels: a reliability diagram (accuracy against
    confidence with the identity reference), a confidence histogram, and the
    per-class distribution of the predicted probability of the positive class.

    Args:
        probabilities: Per-class softmax probabilities, shape ``(n, num_classes)``.
        labels: True class indices, shape ``(n,)``.
        out_path: Where to write the PNG.
        n_bins: Number of confidence bins.
        labels_names: Class names for the legend.
        title: Figure suptitle.

    Returns:
        The :class:`ReliabilityCurve` used to draw the diagram.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    curve = reliability_curve(probabilities, labels, n_bins=n_bins)
    confidence = probabilities.max(axis=1)
    centers = 0.5 * (curve.bin_edges[:-1] + curve.bin_edges[1:])
    width = 1.0 / n_bins

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel 1: reliability diagram.
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="perfect calibration")
    ax.bar(
        centers,
        np.nan_to_num(curve.bin_accuracy),
        width=width * 0.9,
        color="#3b7dd8",
        edgecolor="white",
        label="accuracy",
    )
    gap = np.where(
        np.isnan(curve.bin_accuracy),
        0.0,
        np.nan_to_num(curve.bin_confidence) - np.nan_to_num(curve.bin_accuracy),
    )
    ax.bar(
        centers,
        gap,
        width=width * 0.9,
        bottom=np.nan_to_num(curve.bin_accuracy),
        color="#e06666",
        alpha=0.5,
        edgecolor="white",
        label="confidence gap",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability diagram (ECE = {curve.ece:.3f})")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 2: confidence histogram.
    ax = axes[1]
    ax.hist(confidence, bins=n_bins, range=(0, 1), color="#5aa469", edgecolor="white")
    ax.axvline(
        confidence.mean(), color="#333333", linestyle="--", label=f"mean {confidence.mean():.2f}"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("count")
    ax.set_title("Confidence histogram")
    ax.legend(loc="upper left", fontsize=8)

    # Panel 3: per-class distribution of P(positive).
    ax = axes[2]
    p_pos = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
    for cls, name, color in zip((0, 1), labels_names, ("#d98c5f", "#3b7dd8"), strict=False):
        mask = labels == cls
        if mask.any():
            ax.hist(
                p_pos[mask],
                bins=n_bins,
                range=(0, 1),
                alpha=0.6,
                color=color,
                edgecolor="white",
                label=f"true {name}",
            )
    ax.set_xlim(0, 1)
    ax.set_xlabel("predicted P(positive)")
    ax.set_ylabel("count")
    ax.set_title("Predicted probability by true class")
    ax.legend(loc="upper center", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return curve
