"""
Neural network surrogate analysis of the BBO capstone functions
========================================================================
Trains a small neural network (MLPRegressor) as a surrogate model for each
function's observed data, then:
- Estimates gradients via finite differences on the trained surrogate, to see
  which input dimensions the surrogate is most sensitive to (a numerical
  analogue of what backpropagation computes internally)
- Compares the neural network's fit quality (R^2) against a plain linear
  regression, as a way to quantify how much non-linearity the data seems to
  contain

This is a secondary, exploratory tool -- it does NOT replace the Gaussian
Process used for actual query selection. With only a handful of points per
function, a neural network is prone to overfitting, so results should be read
as rough signals, not precise ground truth.

Requirements: numpy, scikit-learn
Usage: place this file next to bbo_optimizer.py (it reuses its data-loading
logic) and import the functions in the notebook.
"""

import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from bbo_optimizer import _load_history, FUNCTION_DIMS


def _prepare_regression_data(func_id):
    """Load all (initial + weekly) points for a function as (X, y) for regression."""
    X, y = _load_history(func_id)
    return np.array(X), np.array(y)


def train_surrogate(func_id, hidden_layer_sizes=(16, 8), max_iter=3000, random_state=42, alpha=1.0):
    """
    Train a small MLPRegressor as a surrogate model for a function, on
    standardised inputs and outputs. Returns the fitted model and the scalers
    used, so they can be reused for gradient estimation.

    alpha (L2 regularisation strength) defaults to 1.0 rather than sklearn's
    default 0.0001: a hyperparameter experiment on this project's own data
    (Functions 4 and 8) showed the default under-regularises with only 24-44
    points available, causing unstable/overfit fits (e.g. Function 4's CV R^2
    went from -0.100 at alpha=0.0001 to +0.145 at alpha=1.0). Pass a smaller
    alpha explicitly if you have enough data to prefer more flexibility.
    """
    X, y = _prepare_regression_data(func_id)
    n_points = len(y)
    dim = FUNCTION_DIMS[func_id]

    if n_points < 8:
        print(f"Function {func_id}: only {n_points} points, too few to train a reliable surrogate.")
        return None

    y_range = y.max() - y.min()
    if y_range < 0.01:
        print(f"Function {func_id}: output range is only {y_range:.2e} across all {n_points} observed "
              f"points (essentially flat/near-zero) -- no real signal yet to fit a surrogate to.")
        return None

    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=(n_points >= 20),
        validation_fraction=0.2,
    )
    model.fit(X_scaled, y_scaled)
    return model, x_scaler, y_scaler


def surrogate_gradients(func_id, epsilon=1e-3):
    """
    Estimate the surrogate's gradient at the best observed point via finite
    differences: for each input dimension, perturb it slightly and see how
    much the surrogate's predicted output changes. Larger |gradient| suggests
    the surrogate has learned that dimension matters more, at least locally
    around the current best point -- a numerical proxy for what
    backpropagation computes internally during training.
    """
    result = train_surrogate(func_id)
    if result is None:
        return None
    model, x_scaler, y_scaler = result

    X, y = _prepare_regression_data(func_id)
    dim = FUNCTION_DIMS[func_id]
    best_idx = int(np.argmax(y))
    x0 = X[best_idx].copy()

    gradients = np.zeros(dim)
    for d in range(dim):
        x_plus, x_minus = x0.copy(), x0.copy()
        x_plus[d] = min(x_plus[d] + epsilon, 1.0)
        x_minus[d] = max(x_minus[d] - epsilon, 0.0)

        y_plus = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(x_plus.reshape(1, -1))).reshape(-1, 1))[0, 0]
        y_minus = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(x_minus.reshape(1, -1))).reshape(-1, 1))[0, 0]

        denom = x_plus[d] - x_minus[d]
        gradients[d] = (y_plus - y_minus) / denom if denom > 0 else 0.0

    order = np.argsort(np.abs(gradients))[::-1]
    print(f"\nFunction {func_id} ({dim}-D) -- estimated surrogate gradient at best point so far:")
    for rank, idx in enumerate(order, start=1):
        print(f"  #{rank}: x{idx + 1}  gradient={gradients[idx]:+.3f}")
    return gradients


def surrogate_gradients_all():
    """Run gradient estimation for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        surrogate_gradients(func_id)


def compare_linear_vs_nn(func_id, cv_folds=5):
    """
    Compare a plain linear regression against the neural network surrogate
    using cross-validated R^2, to quantify how much non-linear structure the
    data seems to contain. A much higher R^2 for the neural network suggests
    real non-linearity the linear model can't capture; similar R^2 suggests
    the linear model is already a reasonable approximation (at least with the
    current amount of data).
    """
    X, y = _prepare_regression_data(func_id)
    n_points = len(y)
    dim = FUNCTION_DIMS[func_id]

    if n_points < 8:
        print(f"Function {func_id}: only {n_points} points, too few for a reliable comparison.")
        return None

    y_range = y.max() - y.min()
    if y_range < 0.01:
        print(f"\nFunction {func_id} ({dim}-D) -- output range is only {y_range:.2e} across all "
              f"{n_points} observed points (essentially flat/near-zero). R^2 is not meaningful here "
              f"and can blow up numerically on such small-scale data -- both models trivially predict "
              f"~0, since there is no real variation to explain yet.")
        return None

    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    X_scaled = x_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    folds = max(2, min(cv_folds, n_points // 3))

    lin_scores = cross_val_score(LinearRegression(), X_scaled, y_scaled, cv=folds, scoring="r2")

    nn_model = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1.0, max_iter=3000, random_state=42,
                             early_stopping=(n_points >= 20), validation_fraction=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn_scores = cross_val_score(nn_model, X_scaled, y_scaled, cv=folds, scoring="r2")

    print(f"\nFunction {func_id} ({dim}-D) -- {n_points} points, {folds}-fold CV R^2:")
    print(f"  Linear regression -> R^2 = {lin_scores.mean():.3f} (+/- {lin_scores.std():.3f})")

    if np.isnan(nn_scores).any():
        print(f"  Neural network    -> could not be reliably evaluated with only {n_points} points "
              f"(too few for stable training within cross-validation folds)")
        return {"linear_r2": lin_scores.mean(), "nn_r2": None}

    print(f"  Neural network    -> R^2 = {nn_scores.mean():.3f} (+/- {nn_scores.std():.3f})")
    gap = nn_scores.mean() - lin_scores.mean()
    if gap > 0.15:
        print(f"  -> Large gap ({gap:+.3f}): strong evidence of non-linear structure")
    elif gap > 0.05:
        print(f"  -> Moderate gap ({gap:+.3f}): some non-linear structure")
    else:
        print(f"  -> Small gap ({gap:+.3f}): linear model already competitive")

    return {"linear_r2": lin_scores.mean(), "nn_r2": nn_scores.mean()}


def compare_linear_vs_nn_all():
    """Run the linear-vs-neural-network comparison for all 8 functions."""
    for func_id in FUNCTION_DIMS:
        compare_linear_vs_nn(func_id)
