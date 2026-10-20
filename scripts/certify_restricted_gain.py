"""Exact stopped-gain intervals after removing a zero initial exercise right.

The candidate set is fixed at dates 1,...,J. Every reward interval must have a
non-negative lower endpoint and the initial reward must equal [0,0]. The
selected stopping date must belong to the retained calendar. Conditional
increment one cancels from every retained difference, so its row is ignored.

This finite-calendar calculation propagates supplied enclosures. It does not
verify a martingale, the conditional-function enclosures or sampling inference.
Float inputs, when supplied, mean their exact finite binary values. Rational
arithmetic evaluates every retained date span separately.
"""
from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
from pathlib import Path


def _exact(value, name):
    if isinstance(value, bool) or type(value) not in (int, float, Fraction):
        raise ValueError(f'{name} requires a finite exact rational or binary float')
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return Fraction(value)


def _rows(value, name):
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f'{name} must be a full indexed row sequence')
    return value


def _interval(row, name):
    row = _rows(row, name)
    if len(row) != 2:
        raise ValueError(f'{name} requires exactly two endpoints')
    lower, upper = (_exact(row[i], name) for i in range(2))
    if lower > upper:
        raise ValueError(f'{name} has reversed endpoints')
    return lower, upper


def gain_interval_without_zero_initial_right(
    reward_intervals, realised_values, conditional_brackets, stopping_index,
    *, exercise_calendar=None,
):
    """Return a Fraction pair for the maximum over every date 1,...,J.

    All three arrays have length J+1, including the initial date. Reward and
    realised-value rows are finite; conditional rows 2,...,J contain finite
    ordered endpoint pairs. Conditional rows zero and one are placeholders
    and are never accessed. An optional calendar has the same length and
    consists of finite, non-negative, strictly increasing dates.

    Each candidate encloses
      H_j-H_tau-sum(v_r)+sum(c_r),   j>tau,
      H_j-H_tau+sum(v_r)-sum(c_r),   j<tau,
    over precisely the indices between j and tau. The selected-date candidate
    equals zero exactly. All retained candidates enter the maximum, including
    candidates whose reward upper endpoint is zero.
    """
    rewards = _rows(reward_intervals, 'Reward rows')
    values = _rows(realised_values, 'Realised-value rows')
    brackets = _rows(conditional_brackets, 'Conditional rows')
    count = len(rewards)
    if count < 2 or len(values) != count or len(brackets) != count:
        raise ValueError('A complete calendar with J>=1 and matching full row arrays is required')
    if type(stopping_index) is not int or not 1 <= stopping_index < count:
        raise ValueError('The selected stopping index must belong to dates 1,...,J')
    if exercise_calendar is not None:
        dates = _rows(exercise_calendar, 'Exercise calendar')
        if len(dates) != count:
            raise ValueError('The exercise calendar must match every row array')
        dates = tuple(_exact(dates[i], 'Exercise date') for i in range(count))
        if any(t < 0 for t in dates) or any(b <= a for a, b in zip(dates, dates[1:])):
            raise ValueError('Exercise dates must be non-negative and strictly increasing')
    rewards = tuple(_interval(rewards[i], f'Reward row {i}') for i in range(count))
    if rewards[0] != (Fraction(0), Fraction(0)):
        raise ValueError('The removed initial reward must equal [0,0] exactly')
    if any(lower < 0 for lower, _ in rewards):
        raise ValueError('Every reward enclosure must have a non-negative lower endpoint')
    values = tuple(_exact(values[i], f'Realised value {i}') for i in range(count))
    required = {i: _interval(brackets[i], f'Conditional row {i}') for i in range(2, count)}
    lower = upper = Fraction(0)
    tau = stopping_index
    for date in range(1, count):
        if date == tau:
            continue
        span = range(min(date, tau) + 1, max(date, tau) + 1)
        realised = sum((values[i] for i in span), Fraction(0))
        conditional_lower = sum((required[i][0] for i in span), Fraction(0))
        conditional_upper = sum((required[i][1] for i in span), Fraction(0))
        base_lower = rewards[date][0] - rewards[tau][1]
        base_upper = rewards[date][1] - rewards[tau][0]
        if date > tau:
            candidate_lower = base_lower - realised + conditional_lower
            candidate_upper = base_upper - realised + conditional_upper
        else:
            candidate_lower = base_lower + realised - conditional_upper
            candidate_upper = base_upper + realised - conditional_lower
        lower = max(lower, candidate_lower)
        upper = max(upper, candidate_upper)
    return lower, upper


DEFAULT_REPOSITORY = Path(__file__).resolve().parents[1]


def _canonical_digest(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _reconstruct_final_brackets(observation, helpers, rights):
    """Rebuild returned intersections after the complete trace was validated."""
    brackets = [(Fraction(0), Fraction(0))]
    if [row['increment'] for row in observation['initial_queries']] != list(range(1, rights)):
        raise ValueError('Every initial conditional increment is required for the retained full trace')
    for row in observation['initial_queries']:
        brackets.append(helpers._replicated_interval(row['returned_certificate'], signed=True))
    for step in observation['global_rounds']:
        for query in step['queries']:
            index = query['increment']
            if helpers._replicated_interval(query['previous_certificate'], signed=True) != brackets[index]:
                raise ValueError('A returned conditional refinement changes its predecessor')
            returned = helpers._replicated_interval(query['returned_certificate'], signed=True)
            intersection = (max(brackets[index][0], returned[0]), min(brackets[index][1], returned[1]))
            if intersection[1] < intersection[0] or helpers._replicated_interval(query['retained_certificate'], signed=True) != intersection:
                raise ValueError('A final conditional bracket is not its recorded non-empty intersection')
            brackets[index] = intersection
    return brackets


def _independent_full_interval(rewards, values, brackets, stopping):
    """Rebuild the complete-calendar maximum from cancelled cumulative sums."""
    count = len(rewards)
    values_prefix = [Fraction(0)]
    lower_prefix = [Fraction(0)]
    upper_prefix = [Fraction(0)]
    for index in range(1, count):
        values_prefix.append(values_prefix[-1] + values[index])
        lower_prefix.append(lower_prefix[-1] + brackets[index][0])
        upper_prefix.append(upper_prefix[-1] + brackets[index][1])
    candidates = []
    for date in range(count):
        if date == stopping:
            candidates.append((Fraction(0), Fraction(0)))
            continue
        lower = rewards[date][0] - rewards[stopping][1]
        upper = rewards[date][1] - rewards[stopping][0]
        if date > stopping:
            realised = values_prefix[date] - values_prefix[stopping]
            lower += -realised + lower_prefix[date] - lower_prefix[stopping]
            upper += -realised + upper_prefix[date] - upper_prefix[stopping]
        else:
            realised = values_prefix[stopping] - values_prefix[date]
            lower += realised - upper_prefix[stopping] + upper_prefix[date]
            upper += realised - lower_prefix[stopping] + lower_prefix[date]
        candidates.append((lower, upper))
    return (max(lo for lo, _ in candidates), max(hi for _, hi in candidates)), candidates[0]


def _audit_fit(record, ledger, retained, helpers, *, fit_index, root_seed,
               expected_paths, expected_rights, expected_dimension=5):
    """Review one complete retained fit; the CLI fixes eight fits and 1,024 paths."""
    import hashlib
    import json
    rights = len(record['model']['exercise_times'])
    dimension = len(record['model']['spots'])
    if (type(fit_index) is not int or type(root_seed) is not int
            or ledger['fit_index'] != fit_index or ledger['root_seed'] != root_seed
            or record['design']['root_seed'] != root_seed or retained['fit_index'] != fit_index
            or retained['root_seed'] != root_seed or rights != expected_rights or dimension != expected_dimension
            or type(ledger['outer_paths']) is not int or ledger['outer_paths'] != expected_paths
            or record['design']['outer_paths'] != expected_paths or len(record['states']) != expected_paths
            or len(record['stopping_indices']) != expected_paths
            or record['design']['spawn_keys'] != [[0], [1], [3]]):
        raise ValueError('A retained fit changes its dimensions, root seed or declared evaluation streams')
    dates = tuple(_exact(value, 'Exercise date') for value in record['model']['exercise_times'])
    if dates[0] != 0 or any(right <= left for left, right in zip(dates, dates[1:])):
        raise ValueError('The retained exercise calendar must be strictly increasing from zero')
    if len(record['coefficients']) != rights or len(record['frozen_policy']) != rights:
        raise ValueError('The retained fit omits a value function or policy date')
    if len(record['knots']) != 4 or any(_exact(value, 'Hinge strike') <= 0 for value in record['knots']):
        raise ValueError('Every positive retained hinge strike is required')
    for coefficients in record['coefficients']:
        if len(coefficients) != 10:
            raise ValueError('A retained hinge function requires every coefficient')
        for value in coefficients:
            _exact(value, 'Frozen hinge coefficient')
    for rule in record['frozen_policy']:
        if len(rule) not in (0, 21):
            raise ValueError('A retained policy rule omits full-state degree-two coefficients')
        for value in rule:
            _exact(value, 'Frozen policy coefficient')
    for path, stop in zip(record['states'], record['stopping_indices']):
        if len(path) != rights or type(stop) is not int or not 1 <= stop < rights:
            raise ValueError('A retained path omits calendar dates or stops at the removed initial right')
        if path[0] != record['model']['spots']:
            raise ValueError('Every retained path must start at its declared spot vector')
        for state in path:
            if len(state) != dimension or any(_exact(value, 'Retained coordinate') <= 0 for value in state):
                raise ValueError('All retained states require matching finite positive coordinates')
    frozen = {key: record[key] for key in ('model', 'knots', 'coefficients', 'frozen_policy')}
    if _canonical_digest(frozen) != retained['frozen_model_policy_values_sha256']:
        raise ValueError('The retained policy, model or hinge function changed')
    coarse = ledger['arms']['coarse_common']
    adaptive = retained['arms']['adaptive']
    observations = coarse['observations']
    if (len(observations) != expected_paths or len(adaptive['observations']) != expected_paths
            or coarse['outer_paths'] != expected_paths):
        raise ValueError('Every paired coarse and retained adaptive path must be present')
    total_full = [Fraction(0), Fraction(0)]
    total_restricted = [Fraction(0), Fraction(0)]
    total_reduction = [Fraction(0), Fraction(0)]
    unchanged = unchanged_lower = unchanged_upper = ruled_out = retained_intersections = restricted_intersections = 0
    restricted_above_target = 0
    max_full_width = max_restricted_width = max_reduction_width = Fraction(0)
    ordered = hashlib.sha256()
    for index, observation in enumerate(observations):
        stop = record['stopping_indices'][index]
        old = adaptive['observations'][index]
        if (type(observation['path_index']) is not int or observation['path_index'] != index
                or observation['stopping_index'] != stop or old['path_index'] != index
                or old['stopping_index'] != stop):
            raise ValueError('A paired path or stopping index differs from its retained input')
        values = helpers._uniform_realised_values(record, index)
        helpers._uniform_check_trace(observation, record, 'coarse_common', values)
        rewards = [helpers._replicated_interval(row) for row in observation['reward_certificates']]
        brackets = _reconstruct_final_brackets(observation, helpers, rights)
        restricted = gain_interval_without_zero_initial_right(rewards, values,
            [None, None, *brackets[2:]], stop, exercise_calendar=dates)
        full, initial = _independent_full_interval(rewards, values, brackets, stop)
        stored_full = helpers._replicated_interval(observation['signed_gap_certificate'])
        old_adaptive = helpers._replicated_interval(old['signed_gap_certificate'])
        if full != stored_full:
            raise ValueError('The independently rebuilt full-calendar gain differs from its stored enclosure')
        if restricted[0] > full[0] or restricted[1] > full[1]:
            raise ValueError('Restricted maximum endpoints exceed complete-calendar endpoints')
        if max(full[0], old_adaptive[0]) > min(full[1], old_adaptive[1]):
            raise ValueError('The retained complete-calendar coarse and adaptive enclosures do not intersect')
        retained_intersections += 1
        # Different evaluators need not intersect. This count is descriptive.
        restricted_intersections += max(restricted[0], old_adaptive[0]) <= min(restricted[1], old_adaptive[1])
        unchanged += full == restricted
        unchanged_lower += full[0] == restricted[0]
        unchanged_upper += full[1] == restricted[1]
        ruled_out += initial[1] < restricted[0]
        reduction = (max(Fraction(0), initial[0] - restricted[1]),
                     max(Fraction(0), initial[1] - restricted[0]))
        widths = (full[1] - full[0], restricted[1] - restricted[0], reduction[1] - reduction[0])
        max_full_width = max(max_full_width, widths[0])
        max_restricted_width = max(max_restricted_width, widths[1])
        max_reduction_width = max(max_reduction_width, widths[2])
        restricted_above_target += widths[1] > Fraction(1, 100)
        for total, interval in ((total_full, full), (total_restricted, restricted), (total_reduction, reduction)):
            total[0] += interval[0]
            total[1] += interval[1]
        proof = dict(path_index=index, stopping_index=stop,
            full_gain=helpers._replicated_interval_record(*full),
            restricted_gain=helpers._replicated_interval_record(*restricted),
            initial_candidate=helpers._replicated_interval_record(*initial),
            gain_reduction=helpers._replicated_interval_record(*reduction))
        ordered.update(json.dumps(proof, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n')
    exact_mean = lambda total: helpers._replicated_interval_record(total[0] / expected_paths, total[1] / expected_paths)
    full_mean = exact_mean(total_full)
    if helpers._replicated_interval(full_mean) != helpers._replicated_interval(coarse['mean_signed_gap_certificate']):
        raise ValueError('The reconstructed full-calendar sample mean differs from its retained arm summary')
    def rational(value):
        return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))
    return dict(fit_index=fit_index, root_seed=root_seed, retained_paths=expected_paths,
        unchanged_full_interval_count=unchanged,
        unchanged_lower_endpoint_count=unchanged_lower,
        unchanged_upper_endpoint_count=unchanged_upper,
        strictly_ruled_out_initial_candidate_count=ruled_out,
        retained_adaptive_full_interval_intersections=retained_intersections,
        restricted_old_adaptive_interval_intersections=restricted_intersections,
        restricted_paths_above_original_width_target=restricted_above_target,
        mean_full_gain_certificate=full_mean,
        mean_restricted_gain_certificate=exact_mean(total_restricted),
        mean_gain_reduction_certificate=exact_mean(total_reduction),
        maximum_full_width=rational(max_full_width), maximum_restricted_width=rational(max_restricted_width),
        maximum_reduction_width=rational(max_reduction_width), ordered_proof_sha256=ordered.hexdigest())


class _AccessedFiles:
    """Retain SHA-256 identities for every source and evidence file accessed."""
    def __init__(self):
        self.files = {}

    @staticmethod
    def identity(path):
        import hashlib
        from pathlib import Path
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)

    def add(self, path, expected=None):
        from pathlib import Path
        path = Path(path).resolve(strict=True)
        key = str(path)
        actual = self.identity(path)
        if expected is not None and actual['sha256'] != expected:
            raise ValueError(f'A declared source or evidence hash differs: {path}')
        if key in self.files and self.files[key] != actual:
            raise ValueError(f'A previously accessed file changed: {path}')
        self.files[key] = actual
        return path

    def read(self, path, expected=None):
        import json
        path = self.add(path, expected)
        def pairs(values):
            result = {}
            for key, value in values:
                if key in result:
                    raise ValueError('A retained JSON object repeats a key')
                result[key] = value
            return result
        def constant(value):
            raise ValueError('A retained JSON file contains a non-finite constant: ' + value)
        with path.open() as stream:
            return json.load(stream, object_pairs_hook=pairs, parse_constant=constant)

    def verify_unchanged(self):
        for path, recorded in self.files.items():
            if self.identity(path) != recorded:
                raise ValueError(f'An accessed file changed during the audit: {path}')


def _source_path(repository, relative):
    from pathlib import PurePosixPath
    if (not isinstance(relative, str) or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts
            or PurePosixPath(relative).as_posix() != relative):
        raise ValueError('Registered sources require canonical relative paths')
    return repository / relative


def audit_directory(directory, output, *, repository=DEFAULT_REPOSITORY, extension_study=None):
    """Review retained traces and write one new private aggregate after all checks."""
    from datetime import datetime, timezone
    import importlib.util
    import json
    from pathlib import Path
    repository = Path(repository).resolve()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing audit; choose a new output file')
    private = (repository / 'data/private').resolve()
    if not directory.is_relative_to(private) or not output.is_relative_to(private):
        raise ValueError('Retained evidence and its new audit must remain inside data/private')
    tracker = _AccessedFiles()
    tracker.add(__file__)
    helper_path = tracker.add(repository / 'paper/verification/revision_results.py')
    spec = importlib.util.spec_from_file_location('restricted_gain_reporting_helpers', helper_path)
    helpers = importlib.util.module_from_spec(spec)
    import sys
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(helpers)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    design = tracker.read(directory / 'design.json')
    design_sha = tracker.files[str((directory / 'design.json').resolve())]['sha256']
    summary = tracker.read(directory / 'summary.json')
    completion = tracker.read(directory / 'assessment_completion.json')
    registered = tracker.read(directory / 'registration_completion.json')
    sources = design['source_sha256']
    if (type(design['fitting_units']) is not int or design['fitting_units'] != 8
            or design['outer_paths_per_fit'] != 1024 or len(design['fits']) != 8
            or summary['completed_fits'] != 8 or summary['retained_fits'] != 8
            or summary['outer_paths_per_fit'] != 1024 or summary['total_retained_paths'] != 8192
            or len(summary['cases']) != 8 or completion['completed_fits'] != 8 or len(sources) != 15
            or any(record['design_sha256'] != design_sha for record in (summary, completion, registered))
            or any(record['source_sha256'] != sources for record in (summary, completion, registered))):
        raise ValueError('The audit requires the unchanged completed eight-fit allocation design')
    tracker.add(directory / 'summary.json', completion['summary_sha256'])
    for name, expected in sources.items():
        tracker.add(_source_path(repository, name), expected)
    evidence = design['source_evidence']
    extension = Path(extension_study or evidence['frozen_extension_directory']).resolve()
    if not extension.is_relative_to(private):
        raise ValueError('Relocated retained inputs must remain inside data/private')
    cases = []
    for index in range(8):
        fit = design['fits'][index]
        prefix = f'fit_{index:02d}/'
        if fit['index'] != index or fit['root_seed'] != 2026100410 + index:
            raise ValueError('The retained fitting-unit order or root seeds changed')
        folder = directory / prefix
        if (folder / 'assessment_failure.json').exists():
            raise ValueError('A failed fitting unit cannot be treated as a complete retained fit')
        attempt = tracker.read(folder / 'assessment_attempt.json')
        done = tracker.read(folder / 'assessment_completion.json')
        expected_input = evidence['frozen_extension_files'][prefix + 'inputs.json']
        expected_retained = evidence['frozen_extension_files'][prefix + 'result.json']
        context = dict(fit_index=index, root_seed=fit['root_seed'], arm_order=fit['arm_order'],
                       design_sha256=design_sha, source_sha256=sources,
                       retained_input_sha256=expected_input, retained_result_sha256=expected_retained)
        if any(any(record.get(key) != value for key, value in context.items()) for record in (attempt, done)):
            raise ValueError('A retained fit attempt or completion changes its source bindings')
        ledger = tracker.read(folder / 'result.json', done['result_sha256'])
        record = tracker.read(extension / prefix / 'inputs.json', expected_input)
        retained = tracker.read(extension / prefix / 'result.json', expected_retained)
        if (any(ledger.get(key) != value for key, value in context.items())
                or list(ledger['arms']) != fit['arm_order']
                or set(ledger['arms']) != {'cold_common', 'coarse_common', 'adaptive'}
                or record['extension_design_sha256'] != evidence['frozen_extension_design_sha256']
                or record['extension_source_sha256'] != evidence['frozen_extension_source_sha256']
                or record['extension_fit_index'] != index or record['frozen_model_policy_values_sha256'] != retained['frozen_model_policy_values_sha256']
                or retained['input_sha256'] != expected_input or retained['design_sha256'] != evidence['frozen_extension_design_sha256']
                or retained['source_sha256'] != evidence['frozen_extension_source_sha256']):
            raise ValueError('A retained result or relocated input changes its registered context')
        compact = {key: value for key, value in ledger.items() if key != 'arms'}
        compact['arms'] = {name: {key: value for key, value in arm.items() if key != 'observations'}
                           for name, arm in ledger['arms'].items()}
        if compact != summary['cases'][index]:
            raise ValueError('A primary allocation summary differs from its full result ledger')
        row = _audit_fit(record, ledger, retained, helpers, fit_index=index, root_seed=fit['root_seed'],
                         expected_paths=1024, expected_rights=10)
        row.update(retained_input_sha256=expected_input, retained_result_sha256=expected_retained,
                   allocation_result_sha256=done['result_sha256'])
        cases.append(row)
        del ledger, record, retained, compact
    interval = helpers._replicated_interval
    interval_record = helpers._replicated_interval_record
    aggregate = {}
    for name in ('mean_full_gain_certificate', 'mean_restricted_gain_certificate', 'mean_gain_reduction_certificate'):
        bounds = [interval(row[name]) for row in cases]
        aggregate[name] = interval_record(sum(lo for lo, _ in bounds) / 8, sum(hi for _, hi in bounds) / 8)
    def rational(value):
        return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))
    for name in ('maximum_full_width', 'maximum_restricted_width', 'maximum_reduction_width'):
        aggregate[name] = rational(max(helpers._replicated_rational(row[name]) for row in cases))
    for name in ('unchanged_full_interval_count', 'unchanged_lower_endpoint_count', 'unchanged_upper_endpoint_count',
                 'strictly_ruled_out_initial_candidate_count',
                 'retained_adaptive_full_interval_intersections', 'restricted_old_adaptive_interval_intersections',
                 'restricted_paths_above_original_width_target'):
        aggregate[name] = sum(row[name] for row in cases)
    result = dict(schema_version=1, author='Amanjeet Singh', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        completed_fits=8, retained_paths=8192, retained_paths_per_fit=1024,
        design_sha256=design_sha, source_sha256=sources,
        audit_source_sha256=tracker.files[str(Path(__file__).resolve())]['sha256'],
        reporting_helper_sha256=tracker.files[str(helper_path.resolve())]['sha256'],
        cases=cases, aggregate=aggregate, accessed_files_sha256=tracker.files,
        all_accessed_files_unchanged=True, all_coarse_traces_validated=True,
        all_full_gain_reconstructions_match=True, all_endpoint_dominance_checks_pass=True,
        conditional_integrals_rerun=False, new_paths_generated=False, new_fitting_executed=False,
        sampling_coverage_certified=False, policy_improvement_claim=False, cpu_saving_claim=False,
        scope='Exact reconstruction from completed retained coarse-common traces. Removing the zero initial right changes the dual evaluator, not the exercise policy. Numerical sample-mean enclosures do not identify population expectations or actual policy shortfall. Different restricted and complete-calendar gains need not have intersecting enclosures. Width dominance is not implied by endpoint dominance. No conditional oracle, simulation or fitting is rerun.')
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    tracker.verify_unchanged()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(payload)
    return result


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Audit retained gains after removing an exactly zero initial right.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    command = subparsers.add_parser('audit', help='Reconstruct retained intervals without numerical oracle execution')
    command.add_argument('--directory', type=Path, required=True, help='Completed private uniform allocation study')
    command.add_argument('--output', type=Path, required=True, help='New private JSON audit; existing files are preserved')
    command.add_argument('--extension-study', type=Path, help='Relocated retained inputs with unchanged declared hashes')
    command.add_argument('--repository', type=Path, default=Path(DEFAULT_REPOSITORY), help='Checkout containing the fifteen registered sources and reporting helper')
    args = parser.parse_args()
    result = audit_directory(args.directory, args.output, repository=args.repository, extension_study=args.extension_study)
    print('Checked all eight retained fits and 8,192 restricted gain intervals; no numerical oracle execution.')
    print('Exact mean restricted gain:', result['aggregate']['mean_restricted_gain_certificate'])


if __name__ == '__main__':
    main()
