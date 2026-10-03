"""Independent exact-fixture checks for the global residual certificate.

Run with ~/.venvs/qf-certification/bin/python scripts/test_certified_residual.py.
Only the standard library and python-flint are required. These tests generate
no market paths and make no empirical coverage or original-data claims.
"""

from copy import deepcopy
from contextlib import contextmanager, redirect_stdout
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from flint import arb, ctx, fmpq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certify_residual_envelope as certificate


F = Fraction


def independent_ball(value):
    """Convert a test-oracle rational without the implementation's helper."""
    value = value if isinstance(value, F) else F(value)
    return arb(fmpq(value.numerator, value.denominator))


def rational_endpoints(value):
    return F(str(value.lower().fmpq())), F(str(value.upper().fmpq()))


def put_expectation(strike, state, sd):
    """Independent erf representation of a mean-one lognormal put."""
    k, y = independent_ball(strike), independent_ball(state)
    if not state:
        return k
    d = (k/y).log()/sd
    root_two = arb(2).sqrt()
    return (k*(1+((d+sd/2)/root_two).erf())
            - y*(1+((d-sd/2)/root_two).erf()))/2


def hinge_gamma(strike, state, sd):
    if not state:
        return arb(0)
    k, y = independent_ball(strike), independent_ball(state)
    z = (k/y).log()/sd-sd/2
    return (-z*z/2).exp()/(y*sd*(2*arb.pi()).sqrt())


def constant(value):
    return {"knots": [0., 4.], "values": [value, value], "tail_slope": 0.}


def model_fixture():
    """Three exact residuals of one, including a before/after cash edge."""
    return {
        "case": 0,
        "approximation_fingerprint": "exact-constant-fixture",
        "model": {"volatility": 0., "premium_factor": .5,
                  "dividends": [{"ex_time": .5, "discounted_cash": .25}]},
        "option": {"kind": "put", "strike": 2., "maturity": 1.},
        "nodes": [
            {"time": 0., "before": None, "after": constant(3.)},
            {"time": .5, "before": constant(2.), "after": constant(1.)},
            {"time": 1., "before": None, "after": constant(0.)},
        ],
        "decisions": [
            {"time": .5, "phase": "before", "settlement_factor": 1.},
            {"time": .5, "phase": "after", "settlement_factor": .75},
            {"time": 1., "phase": "after", "settlement_factor": .5},
        ],
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_hex(value, path=''):
    if isinstance(value, dict):
        answer = {}
        for key, item in value.items():
            answer.update(fixture_hex(item, path+'/'+key))
        return answer
    if isinstance(value, list):
        answer = {}
        for index, item in enumerate(value):
            answer.update(fixture_hex(item, path+'/'+str(index)))
        return answer
    return {path: value.hex()} if type(value) is float else {}


@contextmanager
def export_fixture():
    """Create disposable, explicitly synthetic private evidence only."""
    with tempfile.TemporaryDirectory(prefix='certificate_unit_fixture_',
                                     dir=certificate.PRIVATE) as folder:
        directory = Path(folder)
        evidence, source = directory/'evidence.txt', directory/'fixture_source.txt'
        evidence.write_text('Exact analytical fixture. No market observations.\n')
        source.write_text('Source-closure fixture.\n')
        models = []
        for case in range(13):
            row = model_fixture()
            row.update(case=case, core_steps=1, wing_steps=1,
                       compression_tolerance=0., initial_value=3.)
            specification = {key: row[key] for key in certificate.APPROXIMATION_FIELDS}
            fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True,
                                                     allow_nan=False).encode()).hexdigest()
            row.update(approximation_fingerprint=fingerprint,
                       matching_arm_fingerprints={'reference': fingerprint, 'selected': fingerprint},
                       residual_fine_fingerprint=fingerprint,
                       exact_binary64_hex=fixture_hex(specification))
            models.append(row)
        payload = {'schema_version': 1,
                   'approximation_fields': list(certificate.APPROXIMATION_FIELDS),
                   'exact_input_convention': 'Synthetic exact binary-input fixture',
                   'models': models}
        input_path = directory/'models.json'
        input_path.write_text(json.dumps(payload, indent=2)+'\n')
        manifest = {'complete': True, 'original_evidence_unchanged': True,
                    'exact_input_convention': payload['exact_input_convention'],
                    'input_sha256': {str(evidence.relative_to(certificate.ROOT)): digest(evidence)},
                    'source_sha256': {str(source.relative_to(certificate.ROOT)): digest(source)},
                    'artifact_sha256': {'models.json': digest(input_path)}}
        manifest_path = directory/'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        yield directory, input_path, manifest_path


def refresh_fixture_artifact_hash(input_path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    manifest['artifact_sha256']['models.json'] = digest(input_path)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')


class ExactFixtureTests(unittest.TestCase):
    def setUp(self):
        self.previous_precision = ctx.prec
        ctx.prec = 160

    def tearDown(self):
        ctx.prec = self.previous_precision

    def assert_encloses(self, result, lower, upper):
        self.assertLessEqual(result['lower'], lower)
        self.assertGreaterEqual(result['upper'], upper)

    def test_binary_inputs_retain_their_exact_rational_values(self):
        self.assertEqual(certificate.q(.1), F.from_float(.1))
        self.assertNotEqual(certificate.q(.1), F(1, 10))
        self.assertEqual(certificate.q(F(2, 3)), F(2, 3))
        for invalid in [True, False, math.nan, math.inf, -math.inf, '0.1']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                certificate.q(invalid)

    def test_outward_quantisation_respects_both_negative_endpoints(self):
        quantum = F(1, 1 << certificate.ENDPOINT_BITS)
        for target in [F(-1, 3), F(1, 3), F(-1, 1 << 90), F(0)]:
            with self.subTest(target=target):
                value = independent_ball(target)
                exact_lo, exact_hi = rational_endpoints(value)
                lo, hi = certificate.outward(value)
                self.assertLessEqual(lo, exact_lo)
                self.assertGreaterEqual(hi, exact_hi)
                self.assertLess(exact_lo-lo, quantum)
                self.assertLess(hi-exact_hi, quantum)
                self.assertEqual((lo/quantum).denominator, 1)
                self.assertEqual((hi/quantum).denominator, 1)

    def test_piecewise_values_and_zero_volatility_extrema_are_exact(self):
        left = certificate.RationalValue([0, 1, 3], [3, 1, 0])
        right = certificate.RationalValue([0, 2, 3], [2, 1, 0])
        self.assertEqual(left.value(F(3, 2)), F(3, 4))
        self.assertEqual(left.value(100), 0)
        result = certificate.residual_range(left, right, arb(0))
        self.assertEqual((result['lower'], result['upper']), (F(-1, 2), F(1)))
        self.assertTrue(result['zero_volatility'])
        self.assertTrue(result['coverage_complete'])
        self.assertTrue(result['tolerance_met'])

    def test_cash_kink_is_an_exact_sum_not_a_rounded_shift(self):
        x, cash = F.from_float(.1), F.from_float(.2)
        rounded_kink = F.from_float(.1+.2)
        exact_kink = x+cash
        self.assertNotEqual(exact_kink, rounded_kink)
        before = certificate.RationalValue([0, cash, rounded_kink], [x, x, 0])
        after = certificate.RationalValue([0, x], [x, 0])
        result = certificate.linear_range(before, after, cash)
        exact_maximum = x*(rounded_kink-exact_kink)/(rounded_kink-cash)
        self.assertGreater(exact_maximum, 0)
        self.assertIn(exact_kink, result['knots'])
        self.assertEqual(result['lower'], 0)
        self.assertEqual(result['upper'], exact_maximum)
        self.assertEqual(exact_maximum,
                         F(3602879701896397, 129807421463370733947818830987264))

    def test_payoff_kink_is_an_exact_product_not_rounded_multiplication(self):
        strike = factor = F.from_float(.1)
        rounded_threshold = F.from_float(.1*.1)
        value = certificate.RationalValue([0, rounded_threshold], [rounded_threshold, 0])
        result = certificate.defect_range(value, .1, .1)
        self.assertEqual(result['threshold'], strike*factor)
        self.assertNotEqual(result['threshold'], rounded_threshold)
        self.assertEqual(result['lower'], strike*factor-rounded_threshold)
        self.assertEqual(result['upper'], 0)

    def test_volatility_time_uses_exact_time_subtraction_before_sqrt(self):
        sigma, t0, t1 = .3, .1, .3
        actual = certificate.transition_sd(sigma, t0, t1)
        rounded = sigma*math.sqrt(t1-t0)
        with ctx.workprec(384):
            oracle = (independent_ball(F.from_float(sigma))
                      * independent_ball(F.from_float(t1)-F.from_float(t0)).sqrt())
            self.assertTrue(actual.contains(oracle))
            self.assertFalse(actual.contains(independent_ball(F.from_float(rounded))))
        self.assertTrue(certificate.transition_sd(0, t0, t1).is_zero())
        for args in [(-.1, 0, 1), (.1, .5, .5), (.1, 1, .5), (.1, -.1, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                certificate.transition_sd(*args)

    def test_signed_hinge_cash_plateau_expectation_matches_put_spread(self):
        plateau = certificate.RationalValue([0, 1, 3], [2, 2, 0])
        self.assertEqual(plateau.hinges, ((F(1), F(-1)), (F(3), F(1))))
        sd = independent_ball(F(3, 4))
        for state in [F(0), F(1, 4), F(1), F(2), F(5), F(100)]:
            with self.subTest(state=state):
                actual = plateau.expectation(state, sd)
                with ctx.workprec(320):
                    oracle = put_expectation(3, state, sd)-put_expectation(1, state, sd)
                self.assertTrue(actual.overlaps(oracle))

    def test_intrinsic_put_residual_has_known_global_extrema(self):
        put = certificate.RationalValue([0, 2], [2, 0])
        sd = certificate.transition_sd(.5, 0, 1)
        tolerance = F(1, 100000)
        result = certificate.residual_range(put, put, sd, tolerance=tolerance)
        with ctx.workprec(320):
            exact_minimum = -2*(arb(1)/(4*arb(2).sqrt())).erf()
            min_lo, min_hi = rational_endpoints(exact_minimum)
        self.assert_encloses(result, min_lo, F(0))
        self.assertLessEqual(min_hi-result['lower'], tolerance)
        self.assertLessEqual(result['upper'], tolerance)
        self.assertTrue(result['tolerance_met'])
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['tail']['limit'], 0)

    def test_tail_limit_is_included_when_supremum_is_not_attained(self):
        left = certificate.RationalValue([0, 2], [1, 1])
        right = certificate.RationalValue([0, 2], [2, 0])
        result = certificate.residual_range(left, right, independent_ball(F(1, 2)))
        # 1-E[(2-yX)+] increases from -1 to 1, attaining 1 only in the limit.
        self.assertEqual(result['lower'], -1)
        self.assertEqual(result['upper'], 1)
        self.assertEqual(result['tail']['upper'], 1)
        self.assertEqual(result['candidate_sup_lower'], 1)
        self.assertTrue(result['tolerance_met'])

    def test_curvature_covers_zero_and_modes_inside_and_outside_cells(self):
        put = certificate.RationalValue([0, 2], [2, 0])
        sd = independent_ball(F(1, 2))
        curvature = certificate.Curvature(put, sd)
        for lower, upper in [(F(0), F(1, 4)), (F(0), F(4)),
                             (F(2), F(4)), (F(1), F(3, 2))]:
            with self.subTest(cell=(lower, upper)):
                bound = curvature.upper(lower, upper)
                self.assertGreater(bound, 0)
                for j in range(17):
                    state = lower+(upper-lower)*F(j, 16)
                    gamma = hinge_gamma(2, state, sd)
                    self.assertGreaterEqual(bound, rational_endpoints(gamma)[1])
        # The cell [0,4] contains the mode, whose exact maximum is known.
        with ctx.workprec(320):
            maximum = arb(F(1, 4).numerator)/F(1, 4).denominator
            maximum = maximum.exp()/(2*independent_ball(F(1, 2))*(2*arb.pi()).sqrt())
        self.assertGreaterEqual(curvature.upper(F(0), F(4)), rational_endpoints(maximum)[1])

    def test_curvature_with_uncertain_mode_and_signed_weights(self):
        plateau = certificate.RationalValue([0, 1, 3], [2, 2, 0])
        sd = arb('0.5 +/- 0.01')
        curvature = certificate.Curvature(plateau, sd)
        for lower, upper in [(F(0), F(1)), (F(2, 3), F(3, 4)), (F(2), F(4))]:
            bound = curvature.upper(lower, upper)
            for scale in [F(49, 100), F(1, 2), F(51, 100)]:
                for j in range(17):
                    state = lower+(upper-lower)*F(j, 16)
                    # Absolute hinge weights give gamma_1+gamma_3, not a difference.
                    oracle = (hinge_gamma(1, state, independent_ball(scale))
                              + hinge_gamma(3, state, independent_ball(scale)))
                    self.assertGreaterEqual(bound, rational_endpoints(oracle)[1])

    def test_resource_limit_preserves_enclosure_but_not_tolerance_claim(self):
        put = certificate.RationalValue([0, 2], [2, 0])
        sd = independent_ball(F(1, 2))
        for controls in [dict(max_rounds=0), dict(max_points=2)]:
            with self.subTest(controls=controls):
                result = certificate.residual_range(put, put, sd,
                    tolerance=F(1, 10**20), **controls)
                self.assertTrue(result['coverage_complete'])
                self.assertFalse(result['tolerance_met'])
                minimum = -2*(sd/(2*arb(2).sqrt())).erf()
                self.assert_encloses(result, rational_endpoints(minimum)[0], F(0))
                self.assertEqual(result['tail']['limit'], 0)
        extra_knot = certificate.RationalValue([0, 1, 2], [2, 1, 0])
        with self.assertRaises(ValueError):
            certificate.residual_range(extra_knot, put, sd, max_points=2)

    def test_pair_aggregation_includes_cash_phase_and_cancels_common_prefix(self):
        result = certificate.certify_model(model_fixture())
        self.assertEqual([edge['kind'] for edge in result['edges']],
                         ['diffusion', 'cash', 'diffusion'])
        self.assertEqual([(edge['lower'], edge['upper']) for edge in result['edges']],
                         [(F(1), F(1))]*3)
        self.assertEqual([right['vertex'] for right in result['rights']], [1, 2, 3])
        self.assertEqual([(pair['lower'], pair['upper']) for pair in result['pairs']],
                         [(F(-2), F(3, 2)), (F(-2), F(1)), (F(-3, 2), F(1))])
        self.assertEqual(result['gain_bound'], 4)
        self.assertTrue(result['ideal_model_bound_certified'])
        self.assertFalse(result['recorded_floating_point_gains_certified'])
        altered_prefix = model_fixture()
        altered_prefix['nodes'][0]['after'] = constant(100.)
        other = certificate.certify_model(altered_prefix)
        self.assertEqual(result['pairs'], other['pairs'])
        self.assertEqual(result['gain_bound'], other['gain_bound'])

    def test_invalid_state_function_and_refinement_inputs_fail(self):
        for knots, values in [([1, 2], [1, 0]), ([0, 1, 1], [2, 1, 0]),
                              ([0, 1], [1]), ([0, math.nan], [1, 0])]:
            with self.subTest(knots=knots), self.assertRaises(ValueError):
                certificate.RationalValue(knots, values)
        put = certificate.RationalValue([0, 2], [2, 0])
        with self.assertRaises(ValueError):
            put.value(F(-1))
        with self.assertRaises(ValueError):
            certificate.linear_range(put, put, F(-1))
        for sd in [arb(-1), arb('0 +/- 0.1'), arb('nan')]:
            with self.subTest(sd=str(sd)), self.assertRaises(ValueError):
                certificate.residual_range(put, put, sd)
        increasing = certificate.RationalValue([0, 2], [0, 1])
        with self.assertRaises(ValueError):
            certificate.residual_range(put, increasing, arb(1))
        for controls in [dict(tolerance=0), dict(max_rounds=-1),
                         dict(max_rounds=True), dict(max_points=1), dict(max_points=True)]:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                certificate.residual_range(put, put, arb(1), **controls)

    def test_invalid_cash_schedule_and_right_order_fail(self):
        mutations = []
        duplicate = model_fixture()
        duplicate['model']['dividends'].append(deepcopy(duplicate['model']['dividends'][0]))
        mutations.append(duplicate)
        wrong_phase = model_fixture()
        wrong_phase['decisions'][0]['phase'] = 'unknown'
        mutations.append(wrong_phase)
        reverse = model_fixture()
        reverse['decisions'][0:2] = reversed(reverse['decisions'][0:2])
        mutations.append(reverse)
        omitted = model_fixture()
        omitted['model']['dividends'].append({'ex_time': .25, 'discounted_cash': .1})
        mutations.append(omitted)
        wrong_maturity = model_fixture()
        wrong_maturity['option']['maturity'] = 2.
        mutations.append(wrong_maturity)
        for index, specification in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                certificate.certify_model(specification)


class ExportIntegrityTests(unittest.TestCase):
    def test_valid_fixture_closure_and_complete_synthetic_run(self):
        with export_fixture() as (directory, input_path, manifest_path):
            original_input, original_manifest = input_path.read_bytes(), manifest_path.read_bytes()
            raw, payload, manifest_raw = certificate.verify_export(input_path)
            self.assertEqual(raw, original_input)
            self.assertEqual(manifest_raw, original_manifest)
            self.assertEqual(len(payload['models']), 13)
            with redirect_stdout(io.StringIO()):
                result = certificate.run(input_path, directory/'certificate', precision=128)
            self.assertEqual(len(result['cases']), 13)
            self.assertTrue(result['ideal_model_bound_certified'])
            self.assertFalse(result['recorded_floating_point_gains_certified'])
            self.assertFalse(result['coverage_of_reported_intervals_certified'])
            self.assertEqual(input_path.read_bytes(), original_input)
            self.assertEqual(manifest_path.read_bytes(), original_manifest)
            completed = json.loads((directory/'certificate/manifest.json').read_text())
            self.assertTrue(completed['complete'])
            for name, expected in completed['artifact_sha256'].items():
                self.assertEqual(digest(directory/'certificate'/name), expected)

    def test_tampered_input_is_rejected_before_output_creation(self):
        with export_fixture() as (directory, input_path, _):
            input_path.write_bytes(input_path.read_bytes()+b' ')
            destination = directory/'rejected'
            with self.assertRaises(ValueError):
                certificate.run(input_path, destination)
            self.assertFalse(destination.exists())

    def test_refreshed_outer_hash_does_not_hide_fingerprint_or_hex_tampering(self):
        for kind in ['value', 'hex', 'arm_fingerprint', 'case_order']:
            with self.subTest(kind=kind), export_fixture() as (_, input_path, manifest_path):
                payload = json.loads(input_path.read_text())
                row = payload['models'][0]
                if kind == 'value':
                    row['nodes'][0]['after']['values'][0] += .25
                elif kind == 'hex':
                    row['exact_binary64_hex']['/model/volatility'] = float(1).hex()
                elif kind == 'arm_fingerprint':
                    row['matching_arm_fingerprints']['selected'] = '0'*64
                else:
                    payload['models'][0], payload['models'][1] = payload['models'][1], payload['models'][0]
                input_path.write_text(json.dumps(payload, indent=2)+'\n')
                refresh_fixture_artifact_hash(input_path, manifest_path)
                with self.assertRaises(ValueError):
                    certificate.verify_export(input_path)

    def test_tampered_evidence_source_and_escaping_path_are_rejected(self):
        for filename in ['evidence.txt', 'fixture_source.txt']:
            with self.subTest(filename=filename), export_fixture() as (directory, input_path, _):
                (directory/filename).write_text('Altered fixture\n')
                with self.assertRaises(ValueError):
                    certificate.verify_export(input_path)
        with export_fixture() as (_, input_path, manifest_path):
            manifest = json.loads(manifest_path.read_text())
            manifest['input_sha256'] = {'../outside.txt': '0'*64}
            manifest_path.write_text(json.dumps(manifest)+'\n')
            with self.assertRaises(ValueError):
                certificate.verify_export(input_path)

    def test_mid_run_input_change_cannot_produce_completion_manifest(self):
        with export_fixture() as (directory, input_path, _):
            original = certificate.certify_model
            calls = []
            def mutate_once(*args, **kwargs):
                result = original(*args, **kwargs)
                if not calls:
                    input_path.write_bytes(input_path.read_bytes()+b'\n')
                calls.append(1)
                return result
            destination = directory/'incomplete'
            with patch.object(certificate, 'certify_model', side_effect=mutate_once):
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    certificate.run(input_path, destination)
            self.assertTrue((destination/'incomplete.json').exists())
            self.assertFalse((destination/'manifest.json').exists())
            self.assertFalse((destination/'summary.json').exists())

    def test_unresolved_tolerance_cannot_produce_completion_manifest(self):
        with export_fixture() as (directory, input_path, _):
            original = certificate.certify_model
            def unresolved(*args, **kwargs):
                result = original(*args, **kwargs)
                result['tolerance_met'] = False
                return result
            destination = directory/'unresolved'
            with patch.object(certificate, 'certify_model', side_effect=unresolved):
                with self.assertRaises(ArithmeticError):
                    certificate.run(input_path, destination)
            self.assertTrue((destination/'incomplete.json').exists())
            self.assertFalse((destination/'manifest.json').exists())
            self.assertFalse((destination/'summary.json').exists())

    def test_existing_output_is_never_overwritten(self):
        with export_fixture() as (directory, input_path, _):
            destination = directory/'existing'
            destination.mkdir()
            sentinel = destination/'preserve.txt'
            sentinel.write_text('Preserve this exact fixture.\n')
            before = sentinel.read_bytes()
            with self.assertRaises(FileExistsError):
                certificate.run(input_path, destination)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(list(destination.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main(verbosity=2)
