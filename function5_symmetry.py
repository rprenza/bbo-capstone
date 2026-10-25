"""
Function 5 symmetry analysis
========================================================================
Plots the relationship between the number of input coordinates set to 1.0
and the observed output y, for Function 5. Across several weekly queries,
the output appeared to depend only on the COUNT of coordinates at the
extreme value 1.0, not on which specific coordinates were set -- e.g.
x=[0,0,1,1] and x=[1,1,0,0] both returned y=1616.6425.

Requirements: numpy, matplotlib
Usage: place this file next to bbo_optimizer.py and import in the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from bbo_optimizer import _load_history


def plot_function5_symmetry(save=True, tolerance=0.001):
    """
    Scatter plot of "number of coordinates == 1.0" vs observed output y for
    Function 5, using all data (initial + weekly). Points with the same count
    are jittered slightly on the x-axis so repeated values are visible rather
    than overlapping exactly.
    """
    X, y = _load_history(5)
    X, y = np.array(X), np.array(y)

    counts = (X >= (1.0 - tolerance)).sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))

    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.08, 0.08, size=len(counts))

    colors = ["gray" if c == 0 else "crimson" for c in counts]
    is_weekly_like = counts > 0  # count>0 points happen to all be weekly queries here

    ax.scatter(counts[counts == 0] + jitter[counts == 0], y[counts == 0],
               alpha=0.5, color="gray", label="Initial data (0 coords at 1.0)", s=50)
    for c in sorted(set(counts[counts > 0])):
        mask = counts == c
        ax.scatter(counts[mask] + jitter[mask], y[mask],
                   color="crimson", s=120, edgecolor="k", zorder=3)

    # Annotate the distinct weekly points with their exact y value
    for c, yv in zip(counts[counts > 0], y[counts > 0]):
        ax.annotate(f"{yv:.1f}", (c, yv), textcoords="offset points",
                   xytext=(8, 4), fontsize=9, color="crimson")

    ax.scatter([], [], color="crimson", s=120, edgecolor="k", label="Weekly queries (coords at 1.0)")

    ax.set_xlabel("Number of input coordinates set to 1.0")
    ax.set_ylabel("Output y")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_title("Function 5 -- output depends on COUNT of coordinates at 1.0,\nnot which specific ones (symmetry pattern)")
    ax.legend()
    plt.tight_layout()

    if save:
        os.makedirs("eda_plots", exist_ok=True)
        path = "eda_plots/function_5_symmetry.png"
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()

    print("\nSummary by count of coordinates at 1.0:")
    for c in sorted(set(counts)):
        vals = y[counts == c]
        print(f"  count={c}: n={len(vals)}, y range=[{vals.min():.2f}, {vals.max():.2f}], mean={vals.mean():.2f}")
