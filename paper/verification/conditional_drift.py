"""Exact finite-tree checks for the paper's compensation argument.

All quantities are declared analytical fixtures. No market data or random draws
are used. Standard-library Fraction arithmetic separates algebraic checks from
floating-point behaviour. See Appendix A and references.bib for attribution.
"""

from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
import json

Node = tuple[int, ...]
Policy = frozenset[Node]


@dataclass(frozen=True)
class Tree:
    """Binary filtration with positive rational transition probabilities."""

    up: tuple[Q, ...]

    def __post_init__(self):
        if not self.up or any(type(p) is not Q or not 0 < p < 1 for p in self.up):
            raise ValueError('Use at least one strictly interior rational probability.')

    @property
    def horizon(self):
        return len(self.up)

    def nodes(self, depth):
        return tuple(product((-1, 1), repeat=depth))

    def probability(self, node):
        result = Q(1)
        for step, direction in enumerate(node):
            result *= self.up[step] if direction == 1 else 1 - self.up[step]
        return result

    def expectation(self, node, values):
        p = self.up[len(node)]
        return (1 - p) * values[node + (-1,)] + p * values[node + (1,)]

    def policies(self):
        """Enumerate stopping antichains, allowing exercise only at 1,...,J."""
        def below(node):
            yield frozenset((node,))
            if len(node) < self.horizon:
                for left, right in product(below(node + (-1,)), below(node + (1,))):
                    yield left | right
        return tuple(left | right for left, right in product(below((-1,)), below((1,))))

    def validate_values(self, values):
        required = {n for j in range(self.horizon + 1) for n in self.nodes(j)}
        if set(values) != required or any(type(x) is not Q for x in values.values()):
            raise ValueError('Supply one rational adapted value at every node.')

    def validate_policy(self, policy):
        rights = {n for j in range(1, self.horizon + 1) for n in self.nodes(j)}
        if not policy <= rights:
            raise ValueError('Policy contains an ineligible exercise right.')
        for leaf in self.nodes(self.horizon):
            if sum(leaf[:j] in policy for j in range(1, self.horizon + 1)) != 1:
                raise ValueError('Policy must select exactly one prefix on every path.')

    def value(self, reward, policy):
        self.validate_policy(policy)
        return sum((self.probability(n) * reward[n] for n in policy), Q(0))

    def optimum(self, reward):
        """Backward recursion, independent of enumerated stopping policies."""
        self.validate_values(reward)
        values = dict(reward)
        for depth in range(self.horizon - 1, 0, -1):
            for node in self.nodes(depth):
                values[node] = max(reward[node], self.expectation(node, values))
        return self.expectation((), values)


@dataclass(frozen=True)
class Metrics:
    value: Q
    gap: Q
    correction: Q
    stopped_compensator: Q


@dataclass
class Audit:
    tree: Tree
    reward: dict[Node, Q]
    penalty: dict[Node, Q]

    def __post_init__(self):
        t = self.tree
        t.validate_values(self.reward)
        t.validate_values(self.penalty)
        if self.penalty[()] != 0:
            raise ValueError('Penalty must start at zero.')
        self.drift = {}
        self.compensator = {(): Q(0)}
        self.norm_budget = Q(0)
        for depth in range(t.horizon):
            for node in t.nodes(depth):
                mu = t.expectation(node, self.penalty) - self.penalty[node]
                self.drift[node] = mu
                for direction in (-1, 1):
                    self.compensator[node + (direction,)] = self.compensator[node] + mu
                if depth >= 1:
                    self.norm_budget += t.probability(node) * abs(mu)
        self.martingale = {n: a - self.compensator[n] for n, a in self.penalty.items()}
        self.optimal_value = t.optimum(self.reward)

    def metrics(self, policy):
        t = self.tree
        t.validate_policy(policy)
        gap = correction = stopped_compensator = Q(0)
        for leaf in t.nodes(t.horizon):
            nodes = tuple(leaf[:j] for j in range(1, t.horizon + 1))
            stopped = next(n for n in nodes if n in policy)
            upper = max(self.reward[n] - self.penalty[n] for n in nodes)
            g = upper - self.reward[stopped] + self.penalty[stopped]
            d = max(self.compensator[n] for n in nodes) - self.compensator[stopped]
            assert g >= 0 and d >= 0
            p = t.probability(leaf)
            gap += p * g
            correction += p * d
            stopped_compensator += p * self.compensator[stopped]
        return Metrics(t.value(self.reward, policy), gap, correction, stopped_compensator)

    def verify(self):
        """Check every admissible policy and every ordered pair exactly."""
        t = self.tree
        assert sum((t.probability(n) for n in t.nodes(t.horizon)), Q(0)) == 1
        for node in self.drift:
            assert t.expectation(node, self.martingale) == self.martingale[node]
        policies = t.policies()
        stats = tuple(self.metrics(p) for p in policies)
        assert max(s.value for s in stats) == self.optimal_value
        for policy, s in zip(policies, stats):
            loss = self.optimal_value - s.value
            assert 0 <= loss <= s.gap + s.correction <= s.gap + self.norm_budget
            # Optional sampling is checked independently by stopped-node weights.
            assert t.value(self.martingale, policy) == 0
            assert t.value(self.penalty, policy) == s.stopped_compensator
        for p, r in product(stats, repeat=2):
            bias = p.stopped_compensator - r.stopped_compensator
            assert p.gap - r.gap == r.value - p.value + bias
            assert abs(bias) <= self.norm_budget
        return len(policies), len(policies) ** 2


def counterexample():
    t = Tree((Q(1, 2), Q(1, 2)))
    reward = {(): Q(0)}
    penalty = {(): Q(0)}
    for depth in (1, 2):
        for node in t.nodes(depth):
            reward[node] = Q(1 if depth == 1 else 1 + node[0])
            penalty[node] = Q(0 if depth == 1 else node[0])
    audit = Audit(t, reward, penalty)
    waiting = frozenset(t.nodes(2))
    optimal = frozenset(((-1,), (1, -1), (1, 1)))
    return audit, waiting, optimal


def fixtures():
    """Deterministic parameter grid under two different probability measures."""
    for probabilities in ((Q(1, 2),) * 3, (Q(1, 3), Q(2, 5), Q(3, 4))):
        t = Tree(probabilities)
        for family in range(3):
            reward = {(): Q(0)}
            for j in range(1, 4):
                for n in t.nodes(j):
                    state = sum(n)
                    choices = (max(2 - state, 0), state**2 - j, 2 + n[0] * state)
                    reward[n] = Q(choices[family])
            for a, b, c in product((-2, 0, 2), repeat=3):
                penalty = {(): Q(0)}
                for j in range(1, 4):
                    for n in t.nodes(j):
                        state = sum(n)
                        penalty[n] = Q(a * state + b * state**2 + c * j * state)
                yield Audit(t, reward, penalty)


def run():
    audit, waiting, optimal = counterexample()
    audit.verify()
    w, o = audit.metrics(waiting), audit.metrics(optimal)
    assert w.gap == o.gap == 0
    assert audit.optimal_value - w.value == w.correction == Q(1, 2)
    assert w.stopped_compensator - o.stopped_compensator == Q(-1, 2)
    cases = policies = pairs = 0
    for case in fixtures():
        n, m = case.verify()
        cases += 1
        policies += n
        pairs += m
    return {
        'scope': 'Declared finite analytical fixtures; no market observations',
        'arithmetic': 'exact rational',
        'parameter_grid_cases': cases,
        'policy_checks_in_grid': policies,
        'ordered_pair_checks_in_grid': pairs,
        'counterexample': {
            'true_loss': str(audit.optimal_value - w.value),
            'apparent_gap': str(w.gap),
            'expected_correction': str(w.correction),
            'coarse_drift_budget': str(audit.norm_budget),
            'true_paired_gain': str(o.value - w.value),
            'apparent_paired_gain': str(w.gap - o.gap),
        },
    }


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Run without optimisation so verification assertions are active.')
    print(json.dumps(run(), indent=2))
