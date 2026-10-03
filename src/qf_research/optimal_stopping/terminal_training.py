"""Train earlier exercise decisions against an integrated terminal reward.

The regression recursion is Longstaff-Schwartz, as implemented in funded_policy.
Only its terminal target changes. The extra interval is integrated analytically
and its known dominated exercise decisions require no additional regression.
"""

from dataclasses import asdict, dataclass
import math
import numpy as np

from .contracts import positive_integer
from .expiry_extension import check_extension, tail_put
from .funded_policy import (
    FundedPolicy, ContinuationRule, _digest, _fit, _validate_paths, fit_policy,
)


@dataclass(frozen=True)
class TerminalTrainedPolicy:
    policy: FundedPolicy
    extension: float

    @property
    def fingerprint(self):
        return _digest(asdict(self))


def fit_terminal_policy(option, paths, degree, extension, *, minimum_itm_paths=32,
                        condition_limit=1e8):
    """Fit on the original path clock, then defer surviving terminal positions.

    The returned policy's final instruction is defined by its extension field.
    Evaluating it as an ordinary old-expiry FundedPolicy would omit that part
    of the strategy. Evaluation must use the terminal-repair construction.
    """
    _validate_paths(paths, option)
    check_extension(paths.model, option, extension)
    if option.kind != 'put' or paths.decisions[-1].phase != 'after':
        raise ValueError('An original post-event terminal put right is required')
    if extension == 0:
        return TerminalTrainedPolicy(fit_policy(option, paths, degree,
            minimum_itm_paths=minimum_itm_paths, condition_limit=condition_limit), 0.)
    degree = positive_integer(degree, 'degree', 0)
    minimum_itm_paths = positive_integer(minimum_itm_paths, 'minimum_itm_paths', 2)
    if isinstance(condition_limit, bool) or not math.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError('Condition limit must be finite and at least one')
    model, rights, states = paths.model, paths.decisions, paths.states
    intrinsic = model.reward(option, rights[-1], states[:, -1])
    cash = np.maximum(intrinsic, tail_put(states[:, -1], option.strike*rights[-1].settlement_factor,
                                         model.volatility, extension))
    rules = [ContinuationRule(0., 1., (), 0, 0, None, None, None, 'terminal') for _ in rights]
    for j in range(len(rights)-2, -1, -1):
        immediate = model.reward(option, rights[j], states[:, j])
        itm = immediate > 0
        rule = _fit(states[itm, j], cash[itm], degree, minimum_itm_paths, condition_limit)
        rules[j] = rule
        if rule.coefficients:
            candidates = np.flatnonzero(itm)
            chosen = candidates[immediate[itm] > rule.predict(states[itm, j])]
            cash[chosen] = immediate[chosen]
    policy = FundedPolicy(model, option, rights, tuple(rules), degree, len(states),
        minimum_itm_paths, float(condition_limit), paths.stream_id)
    return TerminalTrainedPolicy(policy, extension)
