# Model Card — BBO Capstone Optimisation Approach

## Overview

**Name:** BBO Capstone Optimizer

**Type:** Bayesian Optimisation with a Gaussian Process surrogate and an Expected
Improvement acquisition function, implemented as an ask-tell loop with persistent
state. Secondary diagnostic models (SVM classifier, linear regression, small neural
network) are used for interpretation only and play no part in query selection.

**Version:** v1.3, covering all 13 weekly rounds of the challenge.

**Developed by:** Riccardo Prenza, for the Imperial College London Professional
Certificate in Machine Learning and AI capstone project.

**Libraries:** `scikit-optimize` (GP and acquisition function), `scikit-learn`
(diagnostics), `numpy`, `matplotlib`.

## Intended use

**Suitable for**

Optimising expensive black-box functions over a bounded continuous domain where
evaluations are strictly limited, feedback is delayed, and no gradient information is
available. The weekly one-query-per-function structure of this challenge is close to
the real-world setting the method was designed for: simulation calibration, physical
experiments, or any process where each evaluation costs meaningful time or money.

**Use cases to avoid**

This approach should not be used where evaluations are cheap and plentiful — random or
grid search is simpler and competitive there. It should not be used on discrete or
categorical search spaces without modification, since the GP assumes a continuous
domain. And it should not be relied on for high-dimensional problems with a small
budget: the Function 8 results in this project illustrate exactly why, and are discussed
under Limitations.

It is also not a general-purpose predictive model. The surrogate exists to choose the
next query, not to make accurate predictions across the whole domain, and should not be
repurposed as one.

## Details — strategy and how it evolved

**Starting point (Rounds 1–2).** One GP per function, fitted to the course-provided
initial data, with Expected Improvement selecting each query. No initial random or
Sobol sampling, since real observations were already available — spending queries on
blind exploration would have been wasteful given the budget.

**Adding interpretability (Rounds 3–5).** The GP tells you where to query but not why.
I added a diagnostic layer: an SVM classifier on a high/low median split of observed
outputs, comparing linear, RBF and polynomial kernels; and linear-SVM coefficients as a
feature-importance proxy. Function 6 showed RBF at 0.88 cross-validated accuracy against
0.71 for linear, strong evidence of genuine non-linearity. For Function 8, coefficients
consistently ranked x3, x1 and x7 highest, with x6 near zero.

**Surrogate comparison and tuning (Rounds 6–8).** I added a small neural network
surrogate to quantify non-linearity against a linear baseline via cross-validated R².
A hyperparameter experiment on real project data showed the default L2 regularisation
badly under-regularised at this sample size: Function 4's CV R² was -0.100 with a
(16,8) architecture at alpha=0.0001, +0.391 with a smaller (4,) network, and recovered
to +0.145 at alpha=1.0. I changed the module default to alpha=1.0 as a result.

**Hypothesis-driven querying (Rounds 6–9).** The most productive change. For Function 5,
I noticed that structurally different inputs returned identical outputs — x=[0,0,1,1]
and x=[1,1,0,0] both gave exactly 1616.6425. I overrode the acquisition function for
three consecutive rounds to test specific boundary configurations and map the pattern.
This produced the project's clearest structural finding.

**Convergence management (Rounds 10–13).** With functions diverging into plateaued and
still-improving groups, I stopped applying the same approach uniformly: pure exploitation
where results had flattened, retained exploration where sudden jumps remained plausible.

## Performance

**Metric used.** Best observed output per function, tracked round by round, plus a
percentile comparison of each weekly result against the distribution of that function's
initial data. There is no ground truth — the global optimum of each function is
unknown — so all performance claims are relative to what was previously observed.

**Results across the eight functions**

| Function | Dim | Best initial | Best achieved | Pattern |
|---|---|---|---|---|
| 1 | 2D | 0.0000 | 1.501 | Flat for 8 rounds, then a sharp isolated jump |
| 2 | 2D | 0.611 | 0.665 | Early gain, then oscillation |
| 3 | 3D | -0.035 | -0.014 | Steady improvement toward zero |
| 4 | 4D | -4.026 | 0.579 | Large early gain, then a clear plateau |
| 5 | 4D | 1088.9 | 8662.5 | Roughly exponential with boundary coordinates |
| 6 | 5D | -0.714 | -0.136 | Volatile; worst results at domain corners |
| 7 | 6D | 1.365 | 2.675 | Consistent incremental improvement |
| 8 | 8D | 9.598 | 9.982 | Narrow range throughout; marginal gains |

**Notable structural finding.** Function 5's output appears to depend on the number of
input coordinates set to the domain boundary 1.0, rather than on which specific
coordinates are set: approximately 163 with none, 191 with one, 1617 with two, 4441
with three, and 8662 with all four. The jump between one and two coordinates is roughly
eightfold, suggesting a threshold rather than a smooth progression. A midpoint test
(all coordinates at 0.5) returned 32.0 — lower than the all-zeros case — indicating the
function rewards extreme values in either direction and penalises the interior.

## Assumptions and limitations

**Assumptions**

*Deterministic, noise-free functions.* I treated repeated inputs as wasted queries
rather than as noise samples. Function 5 returning exactly 1616.6425 for two
structurally distinct inputs supports this, but it was never confirmed by the course.
If any function is noisy, confidence in single observations is overstated and some
apparent patterns may be chance.

*Domain is exactly [0,1] per dimension.* Inferred from the initial data and the
platform's input format, never explicitly confirmed.

*Stationarity.* A standard GP kernel assumes the function behaves similarly across the
domain. Function 5's sharp threshold at the boundary clearly violates this, which is
why I had to detect that pattern manually rather than rely on the model.

**Limitations and failure modes**

*Budget versus dimensionality.* Thirteen queries is generous in 2D and close to
meaningless in 8D. Function 8's narrow result range is consistent with convergence, but
equally consistent with never having left a small, locally flat region. I cannot
distinguish these explanations, and this is the honest ceiling on what the project can
claim.

*Adaptive sampling bias.* Queries concentrate where results were good, so the collected
data cannot support claims about global function behaviour.

*Exploration cost is real.* Function 7 regressed from 1.980 to 0.759 between rounds
after an early strong result, and Function 4 dropped to -0.568 in Round 3 after setting
a record in Round 2. These are not errors but the price of not overcommitting.

*Manual overrides are not recorded in the data.* The rounds where I rejected the model's
suggestion are documented in my write-ups but not flagged in the JSON history. Someone
reproducing my work from the data alone would not see them.

*Operational fragility.* The ask-tell loop depends on recording the correct result
against the correct input. I twice re-ran a record cell with a previous round's values,
duplicating observations and inflating the query count. Both were caught by checking
state afterwards, and an undo function was added — but the failure mode is real and the
mitigation is procedural, not structural.

## Ethical considerations

The functions here are synthetic, so there are no privacy or fairness concerns in the
usual sense. The relevant considerations are about honesty of reporting.

**Transparency supports reproducibility.** Every observation is persisted to disk, the
code is version-controlled, and suggestions are deterministic given a fixed random seed.
Another researcher can re-run the pipeline and obtain the same proposals. The gap is the
manual overrides, which live in prose rather than data — an inconsistency I would fix in
a future version by adding a provenance flag per observation.

**Transparency supports real-world adaptation.** The clearest practical lesson is that
the surrogate's assumptions should be checked rather than trusted. Documenting where
they were violated (Function 5's non-stationarity) and where the diagnostics disagreed
with each other is more useful to someone adapting this approach than a clean summary
would be.

**Claims are bounded by evidence.** Where the data cannot distinguish between
explanations — particularly for Function 8 — this card says so rather than presenting
the more flattering interpretation. In applied settings, overstating what a model has
learned is the more consequential failure.

## Reflection on this card's structure

Writing this made two gaps visible that I would not have noticed otherwise: that manual
overrides are undocumented in the data, and that Function 8's apparent convergence is
not actually distinguishable from a model stuck in a flat region.

Additional detail that would improve it: per-round provenance flags separating
model-suggested from manually-chosen queries; and an explicit uncertainty estimate from
the GP alongside each best-so-far value, which would make the convergence claims
checkable rather than asserted. The current structure is otherwise sufficient for a
project of this scope — the eight functions are few enough to report individually, which
avoids the aggregation that usually hides a model's failure modes.
