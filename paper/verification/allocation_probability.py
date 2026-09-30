"""Exact finite-law checks for the repeated-training probability appendix.

These are declared mathematical fixtures, not market observations or simulated
research results. Exhaustive enumeration checks selected instances of the
variance identities and a stopped-mean counterexample. It does not prove the
general conditional-probability results or verify a fitted financial model.

Only integers and Fraction objects enter arithmetic. Enumeration is exponential
in the requested counts and is intended for small finite examples. The joint
training law is supplied directly: dependence within a training pair is allowed.
Independent rows and conditionally independent inner evaluations are imposed by
the product construction, rather than inferred from empirical observations.
"""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from typing import Callable, Hashable, Mapping


def _rational(value):
    if type(value) not in (int, Q):
        raise ValueError('Use integers or Fraction objects, without floats or booleans.')
    return Q(value)


def _count(value):
    if type(value) is not int or value <= 0:
        raise ValueError('Counts must be strictly positive integers.')
    return value


@dataclass(frozen=True)
class FiniteLaw:
    """Distinct hashable outcomes with positive exact masses summing to one."""

    atoms: tuple[tuple[Hashable, Q], ...]

    def __post_init__(self):
        atoms = tuple((outcome, _rational(mass)) for outcome, mass in self.atoms)
        if not atoms or any(mass <= 0 for _, mass in atoms):
            raise ValueError('A law requires nonempty support and strictly positive masses.')
        try:
            outcomes = {outcome for outcome, _ in atoms}
        except TypeError as exc:
            raise ValueError('Outcomes must be hashable.') from exc
        if len(outcomes) != len(atoms):
            raise ValueError('Combine repeated outcomes before constructing a law.')
        if sum((mass for _, mass in atoms), Q(0)) != 1:
            raise ValueError('Probability masses must sum exactly to one.')
        object.__setattr__(self, 'atoms', atoms)

    def expectation(self, value: Callable = lambda outcome: outcome):
        return sum((mass * _rational(value(outcome))
                    for outcome, mass in self.atoms), Q(0))

    def variance(self, value: Callable = lambda outcome: outcome):
        mean = self.expectation(value)
        return self.expectation(lambda outcome: (_rational(value(outcome)) - mean) ** 2)

    def pushforward(self, function: Callable):
        masses = defaultdict(Q)
        for outcome, mass in self.atoms:
            masses[function(outcome)] += mass
        return FiniteLaw(tuple(masses.items()))


def independent_average(law: FiniteLaw, count: int) -> FiniteLaw:
    """Enumerate the complete Cartesian product, then merge equal averages."""
    _count(count)
    values = tuple((_rational(value), mass) for value, mass in law.atoms)
    masses = defaultdict(Q)
    for sample in product(values, repeat=count):
        total, probability = Q(0), Q(1)
        for value, mass in sample:
            total += value
            probability *= mass
        masses[total / count] += probability
    return FiniteLaw(tuple(masses.items()))


def _validate_conditionals(training: FiniteLaw, gains: Mapping[Hashable, FiniteLaw]):
    if set(gains) != {state for state, _ in training.atoms}:
        raise ValueError('Supply exactly one conditional gain law per training state.')
    for law in gains.values():
        if not isinstance(law, FiniteLaw):
            raise ValueError('Each conditional gain law must be a FiniteLaw.')
        # Validate scalar values even when subsequent operations only need masses.
        for value, _ in law.atoms:
            _rational(value)


@dataclass(frozen=True)
class VarianceComponents:
    mean: Q
    a: Q
    beta: Q


def variance_components(training: FiniteLaw,
                        gains: Mapping[Hashable, FiniteLaw]) -> VarianceComponents:
    """Return E[Z], Var(E[Z|D]) and E[Var(Z|D)] from the conditional laws."""
    _validate_conditionals(training, gains)
    return VarianceComponents(
        training.expectation(lambda state: gains[state].expectation()),
        training.variance(lambda state: gains[state].expectation()),
        training.expectation(lambda state: gains[state].variance()),
    )


def repeated_training_mean(training: FiniteLaw,
                           gains: Mapping[Hashable, FiniteLaw],
                           m: int, n: int) -> FiniteLaw:
    """Enumerate m independent rows with n evaluations sharing each row's D.

    Conditional gain laws can arise from a common evaluation path applied to
    both policies. Only the scalar gain is retained here. Neither its two policy
    components nor the components of the joint training state need be independent.
    This construction uses no total-variance formula.
    """
    _count(m)
    _count(n)
    _validate_conditionals(training, gains)
    row_masses = defaultdict(Q)
    for state, training_mass in training.atoms:
        for mean, conditional_mass in independent_average(gains[state], n).atoms:
            row_masses[mean] += training_mass * conditional_mass
    return independent_average(FiniteLaw(tuple(row_masses.items())), m)


def pilot_selected_mean(training: FiniteLaw,
                        gains: Mapping[Hashable, FiniteLaw],
                        allocations: FiniteLaw) -> FiniteLaw:
    """Mix fixed-count laws under a pilot independent of all fresh arrays.

    Each allocation outcome is (m, n). The training and gain laws are held fixed
    across pilot outcomes, so there is no pilot-dependent target in this fixture.
    """
    masses = defaultdict(Q)
    for allocation, pilot_mass in allocations.atoms:
        if type(allocation) is not tuple or len(allocation) != 2:
            raise ValueError('Each allocation must be a pair (m, n).')
        m, n = allocation
        for mean, conditional_mass in repeated_training_mean(training, gains, m, n).atoms:
            masses[mean] += pilot_mass * conditional_mass
    return FiniteLaw(tuple(masses.items()))


def training_fixture(kind: str = 'coupled'):
    """Declared law of D=(D0,D1) and Z=(2*D1-D0)+(1+D0+D1)*X.

    D0 and D1 are binary and X is a fresh independent fair sign. Their joint
    training law is coupled, identical within a pair, or independent. Conditional
    gains therefore have both a varying mean and a varying conditional variance.
    """
    laws = {
        'coupled': (((0, 0), Q(1, 2)), ((0, 1), Q(1, 6)),
                    ((1, 0), Q(1, 6)), ((1, 1), Q(1, 6))),
        'shared': (((0, 0), Q(1, 2)), ((1, 1), Q(1, 2))),
        'independent': tuple((state, Q(1, 4)) for state in product((0, 1), repeat=2)),
    }
    if kind not in laws:
        raise ValueError('Unknown training fixture.')
    training = FiniteLaw(laws[kind])
    signs = FiniteLaw(((-1, Q(1, 2)), (1, Q(1, 2))))
    gains = {}
    for state, _ in training.atoms:
        d0, d1 = state
        gains[state] = signs.pushforward(lambda x: 2*d1 - d0 + (1+d0+d1)*x)
    return training, gains


def cost_stopped_signs() -> FiniteLaw:
    """Return (count, sum, cost) when fair signs cost (3+X)/2 and budget is 2.

    Stop at the first completed observation reaching the threshold. Each cost
    is at least one, so two signs suffice to enumerate the complete experiment.
    """
    paths = FiniteLaw(tuple((path, Q(1, 4)) for path in product((-1, 1), repeat=2)))

    def stop(path):
        total, cost = Q(0), Q(0)
        for count, sign in enumerate(path, 1):
            total += sign
            cost += Q(3 + sign, 2)
            if cost >= 2:
                return count, total, cost
        raise AssertionError('The declared threshold must be reached in two steps.')

    return paths.pushforward(stop)


if __name__ == '__main__':
    import json

    training, gains = training_fixture()
    components = variance_components(training, gains)
    fixed = repeated_training_mean(training, gains, 2, 3)
    pilot = FiniteLaw((((1, 1), Q(1, 3)), ((2, 3), Q(2, 3))))
    selected = pilot_selected_mean(training, gains, pilot)
    stopped = cost_stopped_signs()
    fixed_expected = components.a / 2 + components.beta / 6
    selected_expected = (
        components.a * pilot.expectation(lambda counts: Q(1, counts[0]))
        + components.beta * pilot.expectation(lambda counts: Q(1, counts[0]*counts[1]))
    )
    if fixed.variance() != fixed_expected or selected.variance() != selected_expected:
        raise AssertionError('An enumerated variance identity failed.')
    if (stopped.expectation(lambda state: state[1]) != 0
            or stopped.expectation(lambda state: state[1] / state[0]) != Q(1, 4)):
        raise AssertionError('The stopped-mean counterexample failed.')
    print(json.dumps({
        'scope': 'Declared finite mathematical fixtures only; no market observations.',
        'a': str(components.a), 'beta': str(components.beta),
        'fixed_count_variance': str(fixed.variance()),
        'pilot_selected_variance': str(selected.variance()),
        'stopped_sum_expectation': str(stopped.expectation(lambda state: state[1])),
        'stopped_mean_expectation': str(stopped.expectation(lambda state: state[1] / state[0])),
    }, indent=2))
