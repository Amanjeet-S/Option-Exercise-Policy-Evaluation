"""Independently check the assembly of complete residual certificates.

The audit uses exact Fraction arithmetic and never imports the certifier.
Stored conditional-expectation and curvature enclosures remain authenticated
numerical witnesses from the reviewed ball-arithmetic implementation. They
are not independently re-evaluated by this audit.
"""

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/private'
FIELDS = ('model', 'option', 'decisions', 'nodes', 'core_steps', 'wing_steps',
          'compression_tolerance', 'initial_value')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=pairs_object, parse_constant=invalid)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(base, name):
    require(isinstance(name, str), 'A relative evidence path is required')
    path = (base / name).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), 'Invalid evidence path: ' + name)
    return path


def private(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PRIVATE.resolve()), 'Evidence must remain in data/private')
    return path


def verify_hashes(mapping, base):
    require(isinstance(mapping, dict) and mapping, 'A non-empty hash mapping is required')
    for name, expected in mapping.items():
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Malformed checksum')
        require(digest(resolve(base, name)) == expected, 'Evidence checksum differs: ' + name)


def exact(value):
    require(type(value) in (int, float), 'An exact integer or binary64 input is required')
    require(math.isfinite(value), 'Finite inputs are required')
    return Fraction(value)


def rational(value):
    require(isinstance(value, str), 'A canonical rational witness string is required')
    result = Fraction(value)
    require(str(result) == value, 'Non-canonical rational witness')
    return result


def equals(record, key, expected):
    require(rational(record[key]) == expected, 'Exact witness arithmetic differs: ' + key)


def hex_leaves(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(hex_leaves(child, prefix + '/' + key.replace('~', '~0').replace('/', '~1')))
        return result
    if isinstance(value, list):
        result = {}
        for i, child in enumerate(value):
            result.update(hex_leaves(child, prefix + '/' + str(i)))
        return result
    if type(value) is float:
        require(math.isfinite(value), 'Finite exported binary64 inputs are required')
        return {prefix: value.hex()}
    return {}


class LinearFunction:
    """Evaluate exported interpolants exactly, including the constant ray."""

    def __init__(self, record):
        require(isinstance(record, dict) and exact(record['tail_slope']) == 0,
                'A declared constant tail is required')
        self.x = tuple(exact(v) for v in record['knots'])
        self.y = tuple(exact(v) for v in record['values'])
        require(len(self.x) == len(self.y) >= 2 and self.x[0] == 0
                and all(b > a for a, b in zip(self.x, self.x[1:])),
                'Invalid exported linear function')

    def __call__(self, state):
        require(state >= 0, 'A non-negative state is required')
        if state >= self.x[-1]:
            return self.y[-1]
        right = bisect_left(self.x, state)
        if self.x[right] == state:
            return self.y[right]
        left = right - 1
        weight = (state - self.x[left]) / (self.x[right] - self.x[left])
        return (1 - weight) * self.y[left] + weight * self.y[right]

    def nonincreasing(self):
        return all(b <= a for a, b in zip(self.y, self.y[1:]))


def check_linear_range(record, left, right, cap=Fraction(0)):
    require(cap >= 0, 'A non-negative cash cap is required')
    grid = sorted(set(left.x) | {x + cap for x in right.x} | {Fraction(0)})
    values = [left(x) - right(max(Fraction(0), x - cap)) for x in grid]
    require([rational(x) for x in record['knots']] == grid, 'Exact linear-range knot coverage differs')
    require([rational(y) for y in record['values']] == values, 'Exact linear-range values differ')
    lower, upper = min(values), max(values)
    equals(record, 'lower', lower)
    equals(record, 'upper', upper)
    equals(record, 'tail', left.y[-1] - right.y[-1])
    require(lower <= left.y[-1] - right.y[-1] <= upper, 'Constant ray is absent from the range')
    return lower, upper, len(grid)


def check_diffusion(record, left, right, sigma, t0, t1, tolerance):
    require(t1 > t0 >= 0 and sigma >= 0, 'Invalid exact diffusion interval')
    require(record['coverage_complete'] is True, 'Incomplete state-domain coverage')
    equals(record, 'tolerance', tolerance)
    require(type(record['zero_volatility']) is bool and record['zero_volatility'] == (sigma == 0),
            'The deterministic branch differs from the model volatility')
    if sigma == 0:
        lower, upper, knots = check_linear_range(record['exact_linear_range'], left, right)
        for key, expected in (('lower', lower), ('upper', upper),
                              ('candidate_inf_upper', lower), ('candidate_sup_lower', upper)):
            equals(record, key, expected)
        require(record['tolerance_met'] is True, 'An exact deterministic range must meet tolerance')
        return lower, upper, dict(points=0, cells=0, exact_linear_knots=knots)
    require(left.nonincreasing() and right.nonincreasing(),
            'Monotone cell and ray bounds require non-increasing interpolants')
    sd = record['sd_enclosure']
    require(isinstance(sd, list) and len(sd) == 2, 'A two-endpoint volatility-time enclosure is required')
    sd_lo, sd_hi = map(rational, sd)
    square = sigma * sigma * (t1 - t0)
    require(0 <= sd_lo <= sd_hi and sd_lo*sd_lo <= square <= sd_hi*sd_hi,
            'The volatility-time enclosure misses the exact model value')
    points, cells = record['points'], record['cells']
    require(isinstance(points, list) and len(points) >= 2 and isinstance(cells, list),
            'A complete finite state partition is required')
    grid = [rational(p['state']) for p in points]
    require(grid[0] == 0 and all(b > a for a, b in zip(grid, grid[1:]))
            and set(left.x).issubset(set(grid)) and set(right.x).issubset(set(grid))
            and grid[-1] >= max(left.x[-1], right.x[-1]),
            'Partition gaps, missing knots or incomplete ray coverage')
    require(len(cells) == len(points) - 1, 'Each adjacent state pair requires exactly one cell')
    values = []
    for x, point in zip(grid, points):
        f = left(x)
        mean_lo, mean_hi = rational(point['mean_lower']), rational(point['mean_upper'])
        require(mean_lo <= mean_hi, 'Reversed conditional-expectation enclosure')
        require(mean_lo <= right.y[0] and mean_hi >= right.y[-1],
                'Expectation witness excludes the known function range')
        if x == 0:
            require(mean_lo <= right.y[0] <= mean_hi, 'The exact absorbing atom is excluded')
        equals(point, 'value', f)
        equals(point, 'residual_lower', f - mean_hi)
        equals(point, 'residual_upper', f - mean_lo)
        values.append((f, mean_lo, mean_hi, f - mean_hi, f - mean_lo))
    lower_cells, upper_cells = [], []
    for i, cell in enumerate(cells):
        a, b = grid[i:i+2]
        equals(cell, 'a', a)
        equals(cell, 'b', b)
        gamma = rational(cell['curvature_upper'])
        require(gamma >= 0, 'A non-negative curvature bound is required')
        allowance = gamma * (b - a)**2 / 8
        equals(cell, 'chord_allowance', allowance)
        fa, ma_lo, ma_hi, ra_lo, ra_hi = values[i]
        fb, mb_lo, mb_hi, rb_lo, rb_hi = values[i+1]
        lower = max(fb - ma_hi, min(ra_lo, rb_lo) - allowance)
        upper = min(fa - mb_lo, max(ra_hi, rb_hi) + allowance)
        require(lower <= upper, 'The two mathematical cell enclosures are disjoint')
        equals(cell, 'lower', lower)
        equals(cell, 'upper', upper)
        lower_cells.append(lower)
        upper_cells.append(upper)
    limit = left.y[-1] - right.y[-1]
    tail_lo = min(left.y[-1] - values[-1][2], limit)
    tail_hi = limit
    for key, expected in (('start', grid[-1]), ('lower', tail_lo), ('upper', tail_hi), ('limit', limit)):
        equals(record['tail'], key, expected)
    lower = min(*lower_cells, tail_lo)
    upper = max(*upper_cells, tail_hi)
    candidate_inf = min(*(v[4] for v in values), limit)
    candidate_sup = max(*(v[3] for v in values), limit)
    for key, expected in (('lower', lower), ('upper', upper),
                          ('candidate_inf_upper', candidate_inf), ('candidate_sup_lower', candidate_sup)):
        equals(record, key, expected)
    require(lower <= candidate_inf and candidate_sup <= upper, 'Extremum witnesses are inconsistent')
    met = candidate_inf - lower <= tolerance and upper - candidate_sup <= tolerance
    require(type(record['tolerance_met']) is bool and record['tolerance_met'] == met and met,
            'The stated extremum tolerance is not met')
    require(type(record['rounds']) is int and 0 <= record['rounds'] <= 24,
            'Invalid refinement-round record')
    return lower, upper, dict(points=len(points), cells=len(cells), exact_linear_knots=0)


def check_model(spec, result, tolerance):
    require(type(result['case']) is int and result['case'] == spec['case']
            and result['approximation_fingerprint'] == spec['approximation_fingerprint'],
            'Case or retained approximation identity differs')
    require(result['coverage_complete'] is True and result['tolerance_met'] is True
            and result['ideal_model_bound_certified'] is True
            and result['recorded_floating_point_gains_certified'] is False,
            'Certificate scope or completion flags differ')
    model, option, nodes = spec['model'], spec['option'], spec['nodes']
    sigma, maturity = exact(model['volatility']), exact(option['maturity'])
    strike, premium = exact(option['strike']), exact(model['premium_factor'])
    require(option['kind'] == 'put' and strike > 0 and maturity > 0 and premium > 0 and sigma >= 0,
            'Invalid retained put model')
    declared = [(exact(e['ex_time']), exact(e['discounted_cash'])) for e in model['dividends']]
    require(all(t > 0 and c >= 0 for t, c in declared)
            and all(b[0] > a[0] for a, b in zip(declared, declared[1:])), 'Invalid cash calendar')
    events = {t: c for t, c in declared if t <= maturity}
    require(isinstance(nodes, list) and len(nodes) >= 2, 'Full time-node coverage is required')
    times = [exact(node['time']) for node in nodes]
    require(times[0] == 0 and times[-1] == maturity
            and all(b > a for a, b in zip(times, times[1:])), 'Invalid time partition')
    require({t for t, n in zip(times, nodes) if n['before'] is not None} == set(events)
            and nodes[0]['before'] is None, 'Active cash-event coverage differs')
    edges = result['edges']
    require(len(edges) == len(nodes) - 1 + len(events), 'Edge count differs from the full event calendar')
    previous = LinearFunction(nodes[0]['after'])
    vertices = {(Fraction(0), 'after'): (0, previous)}
    bounds, edge_index = [], 0
    counts = dict(points=0, cells=0, exact_linear_knots=0, diffusion_edges=0, cash_edges=0)
    for i, node in enumerate(nodes[1:], 1):
        t0, t1 = times[i-1], times[i]
        after = LinearFunction(node['after'])
        before = None if node['before'] is None else LinearFunction(node['before'])
        edge = edges[edge_index]
        require(edge['kind'] == 'diffusion', 'Expected chronological diffusion edge')
        equals(edge, 'from_time', t0)
        equals(edge, 'to_time', t1)
        lower, upper, row_counts = check_diffusion(edge, previous, before or after, sigma, t0, t1, tolerance)
        bounds.append((lower, upper))
        for key, value in row_counts.items():
            counts[key] += value
        counts['diffusion_edges'] += 1
        edge_index += 1
        if before is not None:
            vertices[t1, 'before'] = (edge_index, before)
            edge = edges[edge_index]
            require(edge['kind'] == 'cash' and edge['tolerance_met'] is True
                    and edge['coverage_complete'] is True, 'Expected complete cash edge')
            equals(edge, 'from_time', t1)
            equals(edge, 'to_time', t1)
            lower, upper, knots = check_linear_range(edge, before, after, events[t1])
            bounds.append((lower, upper))
            counts['cash_edges'] += 1
            counts['exact_linear_knots'] += knots
            edge_index += 1
        vertices[t1, 'after'] = (edge_index, after)
        previous = after
    rights = result['rights']
    require(len(rights) == len(spec['decisions']) and rights, 'All exercise rights must be present')
    right_bounds, seen = [], set()
    for source, record in zip(spec['decisions'], rights):
        key = (exact(source['time']), source['phase'])
        require(key in vertices and key not in seen, 'Undeclared or repeated exercise vertex')
        seen.add(key)
        vertex, value = vertices[key]
        require(type(record['vertex']) is int and record['vertex'] == vertex
                and record['phase'] == key[1], 'Exercise vertex index differs')
        equals(record, 'time', key[0])
        threshold = strike * exact(source['settlement_factor'])
        require(threshold > 0, 'A positive exact exercise threshold is required')
        equals(record, 'threshold', threshold)
        grid = sorted(set(value.x) | {Fraction(0), threshold})
        defects = [max(threshold - x, Fraction(0)) - value(x) for x in grid]
        require([rational(x) for x in record['knots']] == grid
                and [rational(x) for x in record['values']] == defects,
                'Payoff defect knot coverage or exact rational values differ')
        lower, upper = min(defects), max(defects)
        equals(record, 'lower', lower)
        equals(record, 'upper', upper)
        equals(record, 'tail', -value.y[-1])
        right_bounds.append((vertex, lower, upper))
        counts['exact_linear_knots'] += len(grid)
    require((exact(spec['decisions'][-1]['time']), spec['decisions'][-1]['phase']) == (maturity, 'after')
            and all(b[0] > a[0] for a, b in zip(right_bounds, right_bounds[1:])),
            'Distinct chronological rights ending at maturity are required')
    expected_pairs = [(i, j) for i in range(len(rights)) for j in range(i+1, len(rights))]
    require(len(result['pairs']) == len(expected_pairs), 'Pairwise stopping coverage is incomplete')
    bound = Fraction(0)
    for (i, j), pair in zip(expected_pairs, result['pairs']):
        require(type(pair['earlier']) is int and type(pair['later']) is int
                and (pair['earlier'], pair['later']) == (i, j), 'Stopping pair coverage differs')
        a, b = right_bounds[i], right_bounds[j]
        lower = b[1] - a[2] - sum((r[1] for r in bounds[a[0]:b[0]]), Fraction(0))
        upper = b[2] - a[1] - sum((r[0] for r in bounds[a[0]:b[0]]), Fraction(0))
        require(lower <= upper, 'Reversed stopping-gain interval')
        equals(pair, 'lower', lower)
        equals(pair, 'upper', upper)
        bound = max(bound, abs(lower), abs(upper))
    equals(result, 'gain_bound', bound / premium)
    return dict(case=spec['case'], gain_bound=str(bound / premium),
        approximation_fingerprint=spec['approximation_fingerprint'],
        rights=len(rights), pairs=len(expected_pairs), **counts)


def run(folder, output):
    folder, output = private(folder), private(output)
    require(not output.exists(), 'Use a fresh audit directory')
    manifest_path = folder / 'manifest.json'
    manifest = read(manifest_path)
    require(manifest.get('complete') is True and not (folder / 'incomplete.json').exists(),
            'A completed certificate run is required')
    require(manifest.get('source') == 'scripts/certify_residual_envelope.py',
            'The reviewed residual certifier source is required')
    snapshot = {str(manifest_path.relative_to(ROOT)): digest(manifest_path),
                str(Path(__file__).resolve().relative_to(ROOT)): digest(Path(__file__)),
                'tests/test_audit_certified_residual.py': digest(ROOT / 'tests/test_audit_certified_residual.py')}
    for path_key, hash_key in (('input', 'input_sha256'), ('source', 'source_sha256'),
                               ('export_manifest', 'export_manifest_sha256')):
        path = resolve(ROOT, manifest[path_key])
        require(digest(path) == manifest[hash_key], 'Certificate input or source hash differs')
        snapshot[manifest[path_key]] = manifest[hash_key]
    input_path, export_path = private(resolve(ROOT, manifest['input'])), private(resolve(ROOT, manifest['export_manifest']))
    require(input_path.parent == export_path.parent, 'Input and export manifest directories differ')
    exported = read(export_path)
    require(exported.get('complete') is True and exported.get('original_evidence_unchanged') is True,
            'A completed unchanged-input export is required')
    for group in ('input_sha256', 'source_sha256'):
        verify_hashes(exported[group], ROOT)
        for name, value in exported[group].items():
            require(name not in snapshot or snapshot[name] == value, 'Conflicting inherited checksums')
            snapshot[name] = value
    verify_hashes(exported['artifact_sha256'], input_path.parent)
    require(input_path.name in exported['artifact_sha256'], 'The model input is not an exported artifact')
    for name, value in exported['artifact_sha256'].items():
        snapshot[str((input_path.parent/name).relative_to(ROOT))] = value
    verify_hashes(manifest['artifact_sha256'], folder)
    expected_files = {'summary.json', *(f'case_{i:03d}.json' for i in range(13))}
    require(set(manifest['artifact_sha256']) == expected_files
            and {p.name for p in folder.iterdir()} == expected_files | {'manifest.json'},
            'Certificate artifact coverage differs')
    for name, value in manifest['artifact_sha256'].items():
        snapshot[str((folder/name).relative_to(ROOT))] = value
    payload = read(input_path)
    require(payload.get('schema_version') == 1 and payload.get('approximation_fields') == list(FIELDS)
            and payload.get('exact_input_convention') == exported.get('exact_input_convention'),
            'Export schema or exact-input convention differs')
    specs = payload['models']
    require(isinstance(specs, list) and all(type(row['case']) is int for row in specs)
            and [row['case'] for row in specs] == list(range(13)), 'All thirteen model cases are required')
    summary = read(folder / 'summary.json')
    require([row['case'] for row in summary['cases']] == list(range(13)), 'Summary model coverage differs')
    require(type(summary['precision_bits']) is int and summary['precision_bits'] >= 96
            and type(summary['endpoint_quantisation_bits']) is int
            and summary['endpoint_quantisation_bits'] == 80
            and summary['exact_binary64_financial_inputs'] is True
            and summary['exact_cash_and_payoff_rationals'] is True
            and summary['volatility_time_formed_in_ball_arithmetic'] is True,
            'Numerical precision or exact-input conventions differ')
    require(summary['ideal_model_bound_certified'] is True
            and summary['recorded_floating_point_gains_certified'] is False
            and summary['coverage_of_reported_intervals_certified'] is False
            and summary['market_model_validity_certified'] is False,
            'Summary certification scope differs')
    tolerance = rational(summary['tolerance'])
    require(tolerance > 0, 'A positive declared tolerance is required')
    checks = []
    for spec, summary_row in zip(specs, summary['cases']):
        approximation = {key: spec[key] for key in FIELDS}
        fingerprint = hashlib.sha256(json.dumps(approximation, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(spec['approximation_fingerprint'] == fingerprint
                and spec['matching_arm_fingerprints'] == {'reference': fingerprint, 'selected': fingerprint}
                and spec['residual_fine_fingerprint'] == fingerprint
                and spec['exact_binary64_hex'] == hex_leaves(approximation),
                'Exported approximation fingerprint or binary64 identity differs')
        result = read(folder / f"case_{spec['case']:03d}.json")
        check = check_model(spec, result, tolerance)
        for key in ('case', 'gain_bound', 'approximation_fingerprint', 'diffusion_edges', 'cash_edges', 'cells', 'points'):
            require(summary_row[key] == check[key], 'Certificate summary differs: ' + key)
        checks.append(check)
        print(f"Case {spec['case']}: exact certificate assembly verified", flush=True)
    verify_hashes(snapshot, ROOT)
    report = dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Independent exact-rational verification of complete certificate witness assembly.',
        arithmetic='Python standard-library Fraction; no certifier functions or ball-arithmetic package imported.',
        independently_checked=['input, source and artifact hashes', 'exported fingerprints and binary64 identities',
            'exact volatility-time enclosure inequalities', 'complete finite-cell and unbounded-tail coverage',
            'cell monotone and chord intersections', 'candidate extrema and tolerance',
            'cash shifts, exercise payoff kinks and rational ranges', 'all stopping pairs and premium conversion'],
        conditional_expectation_enclosures_independently_recomputed=False,
        curvature_enclosures_independently_recomputed=False,
        trusted_numerical_witnesses='The recorded conditional-expectation and curvature enclosures originate in the authenticated, mathematically reviewed ball-arithmetic implementation. This audit verifies their assembly, not their transcendental evaluation.',
        recorded_floating_point_gains_certified=False, coverage_of_reported_intervals_certified=False,
        cases=checks, totals={key: sum(row[key] for row in checks) for key in
            ('diffusion_edges', 'cash_edges', 'cells', 'points', 'exact_linear_knots', 'rights', 'pairs')},
        evidence_sha256=snapshot, python=platform.python_version(), original_evidence_unchanged=True,
        publication_clearance=False)
    output.mkdir(parents=True, exist_ok=False)
    pending = output / 'checks.pending.json'
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    verify_hashes(snapshot, ROOT)
    pending.rename(output / 'checks.json')
    return report['totals']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.output), sort_keys=True))
