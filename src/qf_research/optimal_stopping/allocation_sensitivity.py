"""Exact finite-grid allocation sensitivity under specified joint cost scenarios.

Variance components and costs are planning inputs. Feasibility refers only to
those supplied scenarios and a fixed budget, without a runtime guarantee.
"""

from fractions import Fraction
import math


def rational(value, name):
    """Treat an input float as its stored dyadic value, not a decimal estimate."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f'{name} must be a finite real scalar')
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return Fraction(value)


def validate(A, B, scenarios, minimum=64, maximum=1024):
    A, B = rational(A, 'training variance'), rational(B, 'evaluation variance')
    if A <= 0 or B <= 0:
        raise ValueError('This sensitivity requires both variance components to be positive')
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise ValueError('Require ordered positive integer repetition limits')
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError('Require a nonempty scenario mapping')
    normalised, grid = {}, None
    for name, costs in scenarios.items():
        if not isinstance(name, str) or not name or not isinstance(costs, dict) or not costs:
            raise ValueError('Require named nonempty cost scenarios')
        if any(type(n) is not int or n < 1 for n in costs):
            raise ValueError('Evaluation counts must be positive integers')
        if grid is not None and set(costs) != grid:
            raise ValueError('All scenarios must cover the same evaluation grid')
        grid = set(costs)
        normalised[name] = {}
        for n, row in costs.items():
            if not isinstance(row, dict) or set(row) != {'setup', 'pair'}:
                raise ValueError('Each cost requires matched setup and pair entries')
            setup, pair = rational(row['setup'], 'setup'), rational(row['pair'], 'pair')
            if setup < 0 or pair <= 0:
                raise ValueError('Require nonnegative setup and positive pair cost')
            normalised[name][n] = dict(setup=setup, pair=pair)
    return A, B, normalised


def charges(n, r, scenarios, overhead=0):
    """Preserve each measured setup/pair combination when taking its maximum."""
    H = rational(overhead, 'overhead')
    if H < 0 or type(r) is not int or r < 1 or type(n) is not int or n < 1:
        raise ValueError('Require nonnegative overhead and positive integer counts')
    if any(n not in row for row in scenarios.values()):
        raise ValueError('Evaluation count is outside the measured grid')
    return {name: H + row[n]['setup'] + r*row[n]['pair'] for name, row in scenarios.items()}


def candidate(A, B, n, r, scenarios, overhead=0):
    cost = charges(n, r, scenarios, overhead)
    return dict(evaluation_paths=n, repetitions=r, variance=(A+B/n)/r,
                scenario_costs=cost, required_budget=max(cost.values()))


def key(row):
    return row['variance'], row['evaluation_paths'], row['repetitions']


def choose(A, B, budget, scenarios, overhead=0, minimum=64, maximum=1024):
    """Minimise the positive plug-in variance over the joint feasible set.

    Costs remain matched within each scenario. The returned budget is the input
    budget; no reference cost is used to reset it. Equality is feasible exactly.
    Nonpositive variance estimates are rejected, never truncated or imputed.
    """
    A, B, scenarios = validate(A, B, scenarios, minimum, maximum)
    budget, H = rational(budget, 'budget'), rational(overhead, 'overhead')
    if budget < 0 or H < 0:
        raise ValueError('Budget and overhead must be nonnegative')
    candidates = []
    for n in sorted(next(iter(scenarios.values()))):
        limits = {name: (budget-H-row[n]['setup'])//row[n]['pair'] for name, row in scenarios.items()}
        available = min(limits.values())
        r = min(maximum, max(0, available))
        feasible = r >= minimum
        record = dict(evaluation_paths=n, repetitions=r, feasible=feasible,
            unconstrained_integer_limits=limits, capped=available>maximum)
        if r > 0:
            record.update(candidate(A, B, n, r, scenarios, H))
        else:
            record.update(variance=None, scenario_costs=None, required_budget=None)
        candidates.append(record)
    feasible = [row for row in candidates if row['feasible']]
    return dict(status='selected' if feasible else 'no_feasible_allocation', budget=budget,
        overhead=H, selected=min(feasible, key=key) if feasible else None, candidates=candidates)


def frontier(A, B, scenarios, overhead=0, minimum=64, maximum=1024):
    """All exact changes of the optimum as the fixed budget increases.

    Equal-cost candidates are considered together. Variance ties favour the
    smaller evaluation count, then the smaller repetition count. A tie-only
    allocation change is labelled separately from a strict variance decrease.
    """
    A, B, scenarios = validate(A, B, scenarios, minimum, maximum)
    H = rational(overhead, 'overhead')
    if H < 0:
        raise ValueError('Overhead must be nonnegative')
    best_at_cost = {}
    for n in sorted(next(iter(scenarios.values()))):
        for r in range(minimum, maximum+1):
            row = candidate(A, B, n, r, scenarios, H)
            cost = row['required_budget']
            if cost not in best_at_cost or key(row) < key(best_at_cost[cost]):
                best_at_cost[cost] = row
    rows, best = [], None
    for cost in sorted(best_at_cost):
        row = best_at_cost[cost]
        if best is None or key(row) < key(best):
            rows.append(dict(**row, strict_variance_improvement=best is None or row['variance']<best['variance']))
            best = row
    return rows


def evaluate(A, B, n, r, budget, scenarios, overhead=0):
    """Cross-evaluate a frozen allocation without changing its sample counts."""
    A, B, scenarios = validate(A, B, scenarios)
    budget, H = rational(budget, 'budget'), rational(overhead, 'overhead')
    if budget < 0 or H < 0:
        raise ValueError('Budget and overhead must be nonnegative')
    row = candidate(A, B, n, r, scenarios, H)
    row['scenario_feasible'] = {name: cost <= budget for name, cost in row['scenario_costs'].items()}
    row['scenario_budget_excess'] = {name: max(Fraction(0), cost-budget) for name, cost in row['scenario_costs'].items()}
    row['scenario_cost_variance'] = {name: cost*row['variance'] for name, cost in row['scenario_costs'].items()}
    row['jointly_feasible'] = all(row['scenario_feasible'].values())
    return row
