"""European benchmarks and exercise-policy research under constant-coefficient GBM."""

from .analytical import Greeks, black_scholes, black_scholes_greeks
from .contracts import Market, Option
from .finite_difference import FiniteDifferenceResult, finite_difference_price
from .least_squares import ExercisePolicy, PolicyEvaluation, evaluate_policy, train_policy
from .simulation import Estimate, estimate_samples, european_monte_carlo, simulate_gbm
from .tree import TreeResult, binomial_price

__all__ = [
    "Market", "Option", "Greeks", "Estimate", "TreeResult", "black_scholes",
    "black_scholes_greeks", "binomial_price", "estimate_samples",
    "european_monte_carlo", "simulate_gbm", "FiniteDifferenceResult", "finite_difference_price",
    "ExercisePolicy", "PolicyEvaluation", "evaluate_policy", "train_policy",
]
