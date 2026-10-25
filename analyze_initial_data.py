"""
Exploratory Data Analysis (EDA) of the BBO capstone initial datasets
========================================================================
Loads the .npy files provided by the course (initial_data/function_N/) for the
eight black-box functions and produces:
- Summary statistics (shape, X range, y range/mean/std)
- Scatter plots to visualise where the best points are so far
- An input-vs-output correlation view for functions with more than 2 dimensions

Requirements: numpy, matplotlib
Usage: place this file in the same folder as initial_data/ and run
       python analyze_initial_data.py
or import the functions in a Jupyter notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

INITIAL_DATA_DIR = "initial_data"
FUNCTION_DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
OUTPUT_DIR = "eda_plots"


def load_function_data(func_id):
    """Load X (input) and y (output) for a given function."""
    folder = os.path.join(INITIAL_DATA_DIR, f"function_{func_id}")
    X = np.load(os.path.join(folder, "initial_inputs.npy"))
    y = np.load(os.path.join(folder, "initial_outputs.npy"))
    return X, y


def print_summary(func_id):
    """Print summary statistics for a given function."""
    X, y = load_function_data(func_id)
    dim = FUNCTION_DIMS[func_id]
    best_idx = int(np.argmax(y))

    print(f"\n{'='*60}")
    print(f"Function {func_id}  ({dim}-D)")
    print(f"{'='*60}")
    print(f"Number of initial points : {len(y)}")
    print(f"X range per dimension:")
    for d in range(dim):
        print(f"    x{d + 1}: [{X[:, d].min():.4f}, {X[:, d].max():.4f}]")
    print(f"Output y  -> min={y.min():.4f}  max={y.max():.4f}  "
          f"mean={y.mean():.4f}  std={y.std():.4f}")
    print(f"Best point so far        : x={np.round(X[best_idx], 4).tolist()}  "
          f"-> y={y[best_idx]:.4f}")


def summary_all():
    """Print statistics for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        print_summary(func_id)


def plot_function(func_id, save=True):
    """
    Generate a plot for a given function:
    - If 2D: scatter plot x1 vs x2, coloured by y (shows where the high-output
      regions are)
    - If >2D: a grid of scatter plots, one per dimension vs y, to see which
      variables seem to influence the output the most
    """
    X, y = load_function_data(func_id)
    dim = FUNCTION_DIMS[func_id]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=80, edgecolor="k")
        best_idx = int(np.argmax(y))
        ax.scatter(X[best_idx, 0], X[best_idx, 1], c="red", s=200,
                   marker="*", edgecolor="k", label="Best point so far")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Function {func_id} ({dim}-D) — initial points")
        ax.legend()
        plt.colorbar(sc, label="y (output)")
        plt.tight_layout()
    else:
        cols = min(dim, 4)
        rows = int(np.ceil(dim / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.array(axes).reshape(-1)
        for d in range(dim):
            ax = axes[d]
            ax.scatter(X[:, d], y, alpha=0.7, edgecolor="k")
            ax.set_xlabel(f"x{d + 1}")
            ax.set_ylabel("y")
            ax.set_title(f"x{d + 1} vs y")
        for d in range(dim, len(axes)):
            axes[d].axis("off")
        fig.suptitle(f"Function {func_id} ({dim}-D) — each variable vs output")
        plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"function_{func_id}.png")
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()


def plot_all():
    """Generate plots for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        plot_function(func_id)


if __name__ == "__main__":
    summary_all()
    plot_all()
