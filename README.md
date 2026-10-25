# BBO Capstone Project — Black-Box Optimisation Challenge

Imperial College London — Professional Certificate in Machine Learning and AI

## In plain language

Imagine looking for the best spot in a dark room, where you are allowed one measurement
a week and the room may have up to eight dimensions. That is this project. Eight hidden
functions accept a set of numbers and return a single score, with no way to see inside
them. Over thirteen weeks I submitted one query per function and used each result to
decide the next.

Rather than guessing, I built a model that learns the shape of each function from every
measurement so far and proposes where the next one is most worth spending. All eight
functions ended above their starting point, and for one I worked out the rule behind its
behaviour.

## Final results

| Function | Inputs | Best at start | Best reached | Record in week |
|---|---|---|---|---|
| 1 | 2 | 0.0000 | **1.5012** | 11 |
| 2 | 2 | 0.6112 | **0.6650** | 6 |
| 3 | 3 | −0.0348 | **−0.0140** | 3 |
| 4 | 4 | −4.0255 | **0.5792** | 10 |
| 5 | 4 | 1088.86 | **8662.48** | 4 |
| 6 | 5 | −0.7143 | **−0.1358** | 9 |
| 7 | 6 | 1.3650 | **2.6746** | 12 |
| 8 | 8 | 9.5985 | **9.9824** | 10 |

A weekly query beat the starting data on all eight functions. Three records arrived in
rounds 10 to 12, meaning the search was still improving when the budget ran out.

## Documentation

- [Datasheet](DATASHEET.md) — how the query dataset was created, what it contains, and
  how it should and should not be used
- [Model card](MODEL_CARD.md) — the optimisation approach, results across all eight
  functions, assumptions and limitations

## Section 1: Project overview

This repository documents my work on the Black-Box Optimisation (BBO) capstone project.
The challenge involves optimising eight unknown ("black-box") functions of increasing
dimensionality (2D to 8D), where the internal mathematical structure is hidden and can
only be probed by submitting query points and observing the resulting output.

The high-level idea mirrors a real Bayesian Optimisation competition: each function
simulates a realistic scenario where evaluations are expensive or limited (radiation-source
detection, drug discovery trials, chemical process yield, ML hyperparameter tuning), and
the goal is to find inputs that maximise the output using as few queries as possible.

This project supports my transition toward data science and PropTech roles, since
Bayesian Optimisation and surrogate modelling transfer directly to real-world problems
such as tuning predictive models with limited data, or calibrating costly physical and
business processes.

## Section 2: Inputs and outputs

Each function receives an input vector `x` of `d` real values, each in the range
`[0, 1]`, where `d` ranges from 2 to 8. The platform requires inputs formatted as a
dash-separated string to six decimal places, e.g. `0.123456-0.654321` for a 2D function.

Each function returns a single scalar output `y`. All eight are maximisation problems;
where the real-world objective is naturally a minimisation (side effects, cost), the
platform applies a transformation so that higher is always better.

Every function starts with a course-provided initial dataset (10 to 40 points depending
on dimensionality), stored as `.npy` files. One additional query per function is
submitted each week for 13 weeks.

## Section 3: Challenge objectives

Maximise each of the eight functions using a strictly limited query budget: one
submission per function per week across 13 weeks, on top of the initial dataset.

Constraints and limitations:

- No visibility into the functions' internal structure or true global optimum
- Roughly a week between submitting a query and receiving its output, so strategy must
  be planned without immediate feedback
- Increasing dimensionality across functions, which makes the higher-dimensional cases
  dramatically harder to explore with so few queries
- At least one function is known to have two distinct optima, adding the risk of
  converging on a local rather than global maximum

## Section 4: Technical approach

The core method is **Bayesian Optimisation with a Gaussian Process surrogate**
(`scikit-optimize`), with **Expected Improvement** selecting each week's query. I chose
this over random or grid search because it makes efficient use of every query —
critical given the budget — by explicitly modelling uncertainty rather than guessing.

Each function has its own independent GP, initialised with the course-provided data and
updated weekly via an ask-tell interface: `suggest()` proposes the next point,
`record()` incorporates the real output once received. All history persists to disk so
progress survives across sessions.

### Interpretability layer

The GP decides *where* to query but reveals little about *why*. I added three
diagnostic tools that complement rather than replace it:

- **SVM kernel comparison** — classifying observed points as high/low output and
  comparing linear, RBF and polynomial kernels. A large gap favouring a non-linear
  kernel indicates genuine curvature. Function 6 showed 0.88 accuracy with RBF against
  0.71 with linear.
- **Linear-SVM feature importance** — ranking input dimensions by coefficient
  magnitude. For Function 8, x3, x1 and x7 dominate while x6 sits near zero.
- **Neural network surrogate** — comparing cross-validated R² against a linear
  baseline to quantify non-linear structure, plus finite-difference gradient estimates
  at each function's best-known point.

A hyperparameter experiment on real project data showed the library default L2
regularisation under-regularises badly at this sample size, so the module default was
changed to `alpha=1.0`.

### Key finding — Function 5 structure

Function 5's output depends on the *number* of input coordinates set to the domain
boundary 1.0, not on which specific coordinates are set. Two structurally different
inputs, `[0,0,1,1]` and `[1,1,0,0]`, returned exactly the same output (1616.6425).

| Coordinates at 1.0 | Observed output |
|---|---|
| 0 | ~163 |
| 1 | ~191 |
| 2 | ~1617 |
| 3 | ~4441 |
| 4 | 8662 |

A midpoint test (all coordinates at 0.5) returned 32.0 — lower than the all-zeros
case — indicating the function rewards extreme values in either direction and penalises
the interior. The roughly eightfold jump between one and two boundary coordinates
suggests a threshold rather than a smooth progression.

This finding came from deliberately overriding the acquisition function for three
consecutive rounds to test a hypothesis, rather than from the optimiser itself. It is
the clearest example in the project of hypothesis-driven investigation outperforming
blind optimisation.

## Repository structure

```
.
├── BBO_Capstone_Notebook.ipynb   # main working notebook
├── bbo_optimizer.py              # ask-tell Bayesian Optimisation loop
├── analyze_initial_data.py       # EDA of the course-provided datasets
├── svm_analysis.py               # SVM kernel comparison and feature importance
├── nn_surrogate_analysis.py      # neural network surrogate and gradient estimates
├── function5_symmetry.py         # Function 5 structural investigation
├── initial_data/                 # course-provided .npy starting datasets
├── bbo_state/                    # persisted weekly query history (JSON)
├── eda_plots/                    # generated figures
├── DATASHEET.md
├── MODEL_CARD.md
└── LICENSE
```

## Setup

```bash
pip install scikit-optimize scikit-learn numpy matplotlib
```

Place `initial_data/` alongside the notebook, then run the cells in order. The weekly
workflow is: `summary()` to check state, `suggest_all()` to generate query points,
submit them to the portal, then `record()` each returned output and `summary()` again
to confirm.

## Author

Riccardo Prenza — [github.com/rprenza](https://github.com/rprenza)
