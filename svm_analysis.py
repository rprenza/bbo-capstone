"""
SVM analysis of the BBO capstone functions
========================================================================
Uses a Support Vector Machine classifier as a fast, interpretable secondary
check alongside the Gaussian Process model: labels each observed point as
"high" or "low" output (above/below the median seen so far), fits an SVM to
separate the two classes, and compares kernels to get a sense of whether the
function's response surface looks linear or non-linear in the region explored
so far.

This does NOT replace the GP/Bayesian Optimization approach -- it's a
diagnostic tool to build intuition about each function's shape and about
which input dimensions seem to matter most.

Requirements: numpy, scikit-learn
Usage: place this file next to bbo_optimizer.py (it reuses its data-loading
logic) and import the functions in the notebook.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from bbo_optimizer import _load_history, FUNCTION_DIMS


def _prepare_classification_data(func_id):
    """Load all (initial + weekly) points for a function and label them
    high/low based on the median output observed so far."""
    X, y = _load_history(func_id)
    X = np.array(X)
    y = np.array(y)
    median = np.median(y)
    labels = (y > median).astype(int)  # 1 = high output, 0 = low output
    return X, y, labels


def svm_kernel_comparison(func_id, kernels=("linear", "rbf", "poly")):
    """
    Fit an SVM classifier (high vs low output) for a function using different
    kernels, and report cross-validated accuracy for each. Higher accuracy
    with a non-linear kernel (rbf/poly) relative to the linear kernel suggests
    the response surface has real curvature or interaction effects between
    inputs, rather than a simple additive/linear structure.
    """
    X, y, labels = _prepare_classification_data(func_id)
    n_points = len(labels)
    n_classes = len(np.unique(labels))

    print(f"\nFunction {func_id} ({FUNCTION_DIMS[func_id]}-D) -- {n_points} points, "
          f"{n_classes} classes after median split")

    if n_points < 6 or n_classes < 2:
        print("  Not enough data yet for a meaningful SVM comparison (need >= 6 points and both classes).")
        return None

    # Standardise inputs: not strictly necessary since inputs are already in
    # [0, 1], but keeps the SVM well-behaved and comparable across functions.
    X_scaled = StandardScaler().fit_transform(X)
    cv_folds = min(5, n_points // 2, np.bincount(labels).min())
    cv_folds = max(cv_folds, 2)

    results = {}
    for kernel in kernels:
        try:
            clf = SVC(kernel=kernel, C=1.0)
            scores = cross_val_score(clf, X_scaled, labels, cv=cv_folds)
            results[kernel] = scores.mean()
            print(f"  {kernel:8s} kernel -> mean CV accuracy = {scores.mean():.3f} "
                  f"(+/- {scores.std():.3f}, {cv_folds}-fold)")
        except Exception as e:
            print(f"  {kernel:8s} kernel -> could not evaluate ({e})")
    return results


def svm_kernel_comparison_all():
    """Run the kernel comparison for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        svm_kernel_comparison(func_id)


def svm_feature_importance(func_id):
    """
    Fit a linear-kernel SVM and inspect the absolute coefficient magnitude
    per input dimension, as a fast proxy for "which dimensions matter most"
    -- useful for higher-dimensional functions where the GP's reasoning is
    harder to inspect directly.
    """
    X, y, labels = _prepare_classification_data(func_id)
    n_points = len(labels)
    n_classes = len(np.unique(labels))
    dim = FUNCTION_DIMS[func_id]

    if n_points < 6 or n_classes < 2:
        print(f"Function {func_id}: not enough data yet for feature importance.")
        return None

    X_scaled = StandardScaler().fit_transform(X)
    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X_scaled, labels)

    importance = np.abs(clf.coef_[0])
    order = np.argsort(importance)[::-1]

    print(f"\nFunction {func_id} ({dim}-D) -- feature importance (linear SVM coefficients):")
    for rank, idx in enumerate(order, start=1):
        print(f"  #{rank}: x{idx + 1}  |coef|={importance[idx]:.3f}")
    return importance


def svm_feature_importance_all():
    """Run feature importance analysis for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        svm_feature_importance(func_id)


def plot_kernel_comparison(save=True):
    """
    Grouped bar chart comparing linear / RBF / polynomial kernel accuracy
    across all 8 functions, so the pattern (which functions look linear vs
    non-linear) is visible at a glance instead of reading a table of numbers.
    """
    import matplotlib.pyplot as plt
    import os

    kernels = ("linear", "rbf", "poly")
    all_scores = {k: [] for k in kernels}
    labels_x = []

    for func_id in FUNCTION_DIMS:
        results = svm_kernel_comparison(func_id) or {}
        labels_x.append(f"F{func_id}")
        for k in kernels:
            all_scores[k].append(results.get(k, 0))

    x = np.arange(len(labels_x))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, k in enumerate(kernels):
        ax.bar(x + (i - 1) * width, all_scores[k], width, label=k)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Mean CV accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("SVM kernel comparison across all 8 functions")
    ax.legend(title="Kernel")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()

    if save:
        os.makedirs("eda_plots", exist_ok=True)
        path = "eda_plots/svm_kernel_comparison.png"
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()


def plot_feature_importance_all(save=True):
    """
    Grid of horizontal bar charts, one per function, showing input dimensions
    ranked by absolute linear-SVM coefficient -- a quick visual of which
    dimensions matter most for each function.
    """
    import matplotlib.pyplot as plt
    import os

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.reshape(-1)

    for i, func_id in enumerate(FUNCTION_DIMS):
        importance = svm_feature_importance(func_id)
        ax = axes[i]
        if importance is None:
            ax.axis("off")
            continue
        dim = FUNCTION_DIMS[func_id]
        order = np.argsort(importance)
        labels = [f"x{d + 1}" for d in order]
        ax.barh(labels, importance[order], color="steelblue")
        ax.set_title(f"Function {func_id} ({dim}-D)")
        ax.set_xlabel("|coefficient|")

    plt.tight_layout()
    if save:
        os.makedirs("eda_plots", exist_ok=True)
        path = "eda_plots/svm_feature_importance.png"
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()


def plot_decision_boundary(func_id, kernel="rbf", save=True):
    """
    Plot the actual data points (coloured by high/low class) together with the
    SVM's decision boundary. Only meaningful for 2D functions (1 and 2) --
    a decision boundary cannot be faithfully drawn on paper for higher
    dimensions, so this raises an error if called on a function with more
    than 2 input dimensions.
    """
    import matplotlib.pyplot as plt
    import os

    dim = FUNCTION_DIMS[func_id]
    if dim != 2:
        raise ValueError(
            f"Function {func_id} has {dim} dimensions -- decision boundary "
            f"plots are only supported for 2D functions (1 and 2)."
        )

    X, y, labels = _prepare_classification_data(func_id)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clf = SVC(kernel=kernel, C=1.0)
    clf.fit(X_scaled, labels)

    # Build a grid over the actual [0, 1] x [0, 1] input domain
    grid_n = 200
    x1_grid = np.linspace(0, 1, grid_n)
    x2_grid = np.linspace(0, 1, grid_n)
    xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])
    grid_scaled = scaler.transform(grid_points)
    preds = clf.predict(grid_scaled).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx1, xx2, preds, alpha=0.25, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
    sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", edgecolor="k", s=80)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Function {func_id} -- SVM ({kernel}) decision boundary\n(blue=low output, red=high output)")
    plt.tight_layout()

    if save:
        os.makedirs("eda_plots", exist_ok=True)
        path = f"eda_plots/function_{func_id}_svm_boundary.png"
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()
