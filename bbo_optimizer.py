"""
BBO Capstone Optimizer
=======================
Ask-tell Bayesian Optimization helper for the Imperial ML Capstone BBO challenge.

Weekly workflow, for each of the 8 functions:
1. Call `suggest(func_id)` to get the next point to try.
2. Copy the printed string into the "Function N" field on the capstone project
   portal and submit.
3. Once you get the output value back from the platform, call
   `record(func_id, y_value)` to update the model.
4. Repeat. The more queries you make, the better the model learns the shape
   of the function.

State (input/output history) is saved to disk in the bbo_state/ folder, so you
can close and reopen Python without losing progress.

Requirements: pip install scikit-optimize numpy
"""

import json
import os
import numpy as np
from skopt import Optimizer
from skopt.space import Real

STATE_DIR = "bbo_state"
os.makedirs(STATE_DIR, exist_ok=True)

# Folder with the real initial data provided by the course
# (initial_data/function_N/initial_inputs.npy, initial_outputs.npy).
# Extract the "Initial_data_points_starter.zip" here so the structure is:
# initial_data/function_1/, function_2/, etc.
INITIAL_DATA_DIR = "initial_data"

# Dimensionality of each of the 8 functions, as shown on the platform
FUNCTION_DIMS = {
    1: 2,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 6,
    8: 8,
}

# Assumption: domain [0, 1] for every variable, based on the initial data
# provided and the platform's input placeholder ("0.000000-0.000000").
# If you discover the real domain differs, change it here.
BOUNDS = (0.0, 1.0)

# The portal has 13 total submission weeks (Module 12 -> Module 24), one
# submission per week covering all 8 functions together.
# So the budget is 13 total queries PER function, not shared across all of them.
TOTAL_WEEKS = 13

_optimizers = {}
_pending = {}  # func_id -> last suggested point, waiting for a result


def _state_path(func_id):
    return os.path.join(STATE_DIR, f"function_{func_id}.json")


def _load_initial_data(func_id):
    """Load the initial (x, y) points provided by the course for a function, if present."""
    folder = os.path.join(INITIAL_DATA_DIR, f"function_{func_id}")
    x_path = os.path.join(folder, "initial_inputs.npy")
    y_path = os.path.join(folder, "initial_outputs.npy")
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path).tolist()
        y = np.load(y_path).tolist()
        return X, y
    return [], []


def _load_history(func_id):
    """
    Load the history for a function: if no saved state exists yet, initialise
    it with the real initial data provided by the course (initial_data/).
    Subsequent weekly queries are then accumulated on top of this data.
    """
    path = _state_path(func_id)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("X", []), data.get("y", [])

    # First time: start from the real initial data, if available
    X, y = _load_initial_data(func_id)
    if X:
        _save_history(func_id, X, y)
        print(f"Function {func_id}: loaded {len(y)} initial points from {INITIAL_DATA_DIR}/function_{func_id}/")
    return X, y


def _save_history(func_id, X, y):
    with open(_state_path(func_id), "w") as f:
        json.dump({"X": X, "y": y}, f, indent=2)


def _get_optimizer(func_id):
    if func_id not in _optimizers:
        dim = FUNCTION_DIMS[func_id]
        space = [Real(*BOUNDS) for _ in range(dim)]
        X, y = _load_history(func_id)
        # We already have real initial data (from the course or previous
        # queries), so n_initial_points=0: the optimizer uses the GP model
        # right away instead of proposing more random/Sobol points.
        n_initial = 0 if X else 4
        opt = Optimizer(
            dimensions=space,
            base_estimator="GP",
            acq_func="EI",  # Expected Improvement
            acq_optimizer="auto",
            n_initial_points=n_initial,
            initial_point_generator="sobol",
            random_state=42 + func_id,  # different seed per function, otherwise Sobol sequences would coincide across functions
        )
        if X:
            # skopt minimises internally -> we negate y because we want to MAXIMISE
            opt.tell(X, [-v for v in y])
        _optimizers[func_id] = opt
    return _optimizers[func_id]


def suggest(func_id):
    """Return the next point to try for a given function, formatted for the platform."""
    opt = _get_optimizer(func_id)
    x = opt.ask()
    _pending[func_id] = x
    formatted = "-".join(f"{v:.6f}" for v in x)
    print(f"Function {func_id} ({FUNCTION_DIMS[func_id]}-D) -> submit this value:")
    print(formatted)
    return formatted


def record(func_id, y_value, x=None):
    """
    Record the result returned by the platform for the last suggested point
    (or an explicit x, if you're re-entering a past result)."""
    opt = _get_optimizer(func_id)
    x = x if x is not None else _pending.get(func_id)
    if x is None:
        raise ValueError(
            f"No pending suggestion for function {func_id}. "
            f"Call suggest({func_id}) first, or pass x explicitly."
        )
    opt.tell(x, -y_value)  # negated because skopt minimises, we want to maximise

    X, y = _load_history(func_id)
    X.append(list(x))
    y.append(y_value)
    _save_history(func_id, X, y)
    _pending.pop(func_id, None)

    best_idx = int(np.argmax(y))
    print(f"Recorded. Best result so far for function {func_id}: "
          f"y={y[best_idx]:.6f} at x={X[best_idx]}")


def best(func_id):
    """Show the best result found so far for a function."""
    X, y = _load_history(func_id)
    if not y:
        print(f"No data yet for function {func_id}.")
        return None
    idx = int(np.argmax(y))
    print(f"Function {func_id}: best y={y[idx]:.6f} at x={X[idx]}")
    return X[idx], y[idx]


def undo_last(func_id):
    """
    Remove the most recently recorded point for a function -- useful if you
    accidentally recorded a duplicate or wrong result. Refuses to remove one of
    the original initial data points (only removes weekly-recorded entries).
    """
    X, y = _load_history(func_id)
    initial_X, _ = _load_initial_data(func_id)
    n_initial = len(initial_X)
    if len(y) <= n_initial:
        print(f"Function {func_id}: nothing to undo (only initial data present).")
        return
    removed_x, removed_y = X.pop(), y.pop()
    _save_history(func_id, X, y)
    _optimizers.pop(func_id, None)  # force the optimizer to rebuild from the corrected history
    print(f"Function {func_id}: removed last entry x={removed_x} -> y={removed_y}")


def undo_last_all():
    """Remove the most recently recorded point for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        undo_last(func_id)


def summary():
    """Print a summary of all 8 functions: initial points, weekly queries used/remaining, best result."""
    for func_id in FUNCTION_DIMS:
        X, y = _load_history(func_id)
        initial_X, _ = _load_initial_data(func_id)
        n_initial = len(initial_X)
        n_weekly = len(y) - n_initial
        remaining = TOTAL_WEEKS - n_weekly
        if not y:
            print(f"Function {func_id}: no data yet")
        else:
            idx = int(np.argmax(y))
            print(f"Function {func_id}: {n_initial} initial points + {n_weekly}/{TOTAL_WEEKS} weekly queries "
                  f"({remaining} remaining), best y={y[idx]:.6f}")


def suggest_all():
    """Generate the suggested point for all 8 functions at once, since the
    weekly submission requires all functions together."""
    results = {}
    for func_id in FUNCTION_DIMS:
        results[func_id] = suggest(func_id)
    return results


def compare_to_initial(func_id):
    """
    Compare each weekly query's output against the initial dataset's distribution.
    Prints, for every weekly point, its percentile rank relative to the initial y
    values -- e.g. "better than 80% of the initial points" -- so you can judge how
    good a guess was without knowing the true global maximum.
    """
    initial_X, initial_y = _load_initial_data(func_id)
    X, y = _load_history(func_id)
    n_initial = len(initial_y)
    weekly_X, weekly_y = X[n_initial:], y[n_initial:]

    if not weekly_y:
        print(f"Function {func_id}: no weekly queries recorded yet.")
        return

    initial_y_arr = np.array(initial_y)
    print(f"\nFunction {func_id} -- initial data: "
          f"min={initial_y_arr.min():.4f}, max={initial_y_arr.max():.4f}, "
          f"mean={initial_y_arr.mean():.4f}")
    for week_num, (x_val, y_val) in enumerate(zip(weekly_X, weekly_y), start=1):
        percentile = 100 * np.mean(initial_y_arr <= y_val)
        verdict = (
            "better than the best initial point" if y_val > initial_y_arr.max() else
            "worse than every initial point" if y_val < initial_y_arr.min() else
            f"better than {percentile:.0f}% of initial points"
        )
        print(f"  Week {week_num}: x={np.round(x_val, 4).tolist()} -> y={y_val:.6f}  ({verdict})")


def compare_all():
    """Run compare_to_initial for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        compare_to_initial(func_id)


def plot_progress(func_id, save=True):
    """
    Plot the initial data together with the weekly queries made so far, so you can
    see visually whether your weekly points are landing in strong or weak regions.
    - If 2D: scatter plot x1 vs x2, coloured by y. Initial points are circles,
      weekly points are stars (numbered by week).
    - If >2D: one scatter per input dimension vs y, with weekly points highlighted.
    """
    import matplotlib.pyplot as plt
    import os as _os

    initial_X, initial_y = _load_initial_data(func_id)
    X, y = _load_history(func_id)
    dim = FUNCTION_DIMS[func_id]
    n_initial = len(initial_y)
    weekly_X, weekly_y = np.array(X[n_initial:]), y[n_initial:]
    initial_X, initial_y = np.array(initial_X), np.array(initial_y)

    _os.makedirs("eda_plots", exist_ok=True)

    if dim == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(initial_X[:, 0], initial_X[:, 1], c=initial_y, cmap="viridis",
                        s=80, edgecolor="k", label="Initial data")
        if len(weekly_y):
            ax.scatter(weekly_X[:, 0], weekly_X[:, 1], c=weekly_y, cmap="viridis",
                      s=250, marker="*", edgecolor="red", linewidth=1.5, label="Weekly queries")
            for i, (xr, yr) in enumerate(zip(weekly_X, weekly_y), start=1):
                ax.annotate(f"W{i}", (xr[0], xr[1]), textcoords="offset points",
                           xytext=(6, 6), fontsize=9, color="red")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Function {func_id} ({dim}-D) -- initial data vs weekly queries")
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
            ax.scatter(initial_X[:, d], initial_y, alpha=0.6, edgecolor="k", label="Initial")
            if len(weekly_y):
                ax.scatter(weekly_X[:, d], weekly_y, c="red", s=150, marker="*",
                          edgecolor="k", label="Weekly")
            ax.set_xlabel(f"x{d + 1}")
            ax.set_ylabel("y")
            ax.set_title(f"x{d + 1} vs y")
        axes[0].legend()
        for d in range(dim, len(axes)):
            axes[d].axis("off")
        fig.suptitle(f"Function {func_id} ({dim}-D) -- initial data vs weekly queries")
        plt.tight_layout()

    if save:
        path = f"eda_plots/function_{func_id}_progress.png"
        plt.savefig(path, dpi=120)
        print(f"Saved: {path}")
    plt.show()


def plot_progress_all():
    """Generate the progress plot for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        plot_progress(func_id)


def final_summary():
    """
    Print one closing table for the whole project: for each function, the best
    score that was already present in the course-provided starting data, the
    best score reached overall, and which weekly round produced the record.

    Reading the table:
    - "Best initial" is the bar each weekly query had to beat.
    - "Best reached" is the highest score across starting data and weekly queries.
    - "Record week" is the weekly round that produced it, or "initial" if no
      weekly query ever beat the starting data.
    """
    header = (f"{'Func':>4}  {'Dim':>3}  {'Best initial':>14}  "
              f"{'Best reached':>14}  {'Record week':>12}  {'Improved':>9}")
    print(header)
    print("-" * len(header))

    improved_count = 0

    for func_id in FUNCTION_DIMS:
        X, y = _load_history(func_id)
        initial_X, initial_y = _load_initial_data(func_id)
        dim = FUNCTION_DIMS[func_id]

        if not y:
            print(f"{func_id:>4}  {dim:>3}  {'no data yet':>14}")
            continue

        n_initial = len(initial_y)
        best_initial = max(initial_y) if initial_y else float("-inf")
        best_overall = max(y)
        best_idx = int(np.argmax(y))

        # Which round produced the record? Weekly rounds start after the
        # initial points, so index n_initial is week 1.
        if best_idx < n_initial:
            record_week = "initial"
            improved = "no"
        else:
            record_week = str(best_idx - n_initial + 1)
            improved = "yes"
            improved_count += 1

        init_str = f"{best_initial:.4f}" if initial_y else "n/a"
        print(f"{func_id:>4}  {dim:>3}  {init_str:>14}  "
              f"{best_overall:>14.4f}  {record_week:>12}  {improved:>9}")

    print("-" * len(header))
    print(f"Functions where a weekly query beat the starting data: "
          f"{improved_count}/{len(FUNCTION_DIMS)}")


if __name__ == "__main__":
    print("BBO Capstone Optimizer loaded. Available functions:")
    print("  suggest(func_id)         -> get the next point to submit")
    print("  record(func_id, y_value) -> record the platform's result")
    print("  best(func_id)            -> show the best result so far")
    print("  summary()                -> overview of all functions")
