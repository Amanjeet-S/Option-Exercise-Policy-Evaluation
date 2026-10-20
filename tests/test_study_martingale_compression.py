"""Study-binding and selection checks with literal temporary records.

I use only the standard library and deterministic mocked numerical methods.
No production stream, model path, polynomial fit or price oracle is called.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations_with_replacement
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(os.environ.get('QF_TEST_REPOSITORY', Path(__file__).resolve().parents[1])).resolve()
SOURCE = ROOT/'scripts/study_martingale_compression.py'
if not SOURCE.is_file():
    ROOT = Path.cwd().resolve(); SOURCE = ROOT/'scripts/study_martingale_compression.py'
specification = importlib.util.spec_from_file_location('compression_study_guard_fixture', SOURCE)
study = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = study
specification.loader.exec_module(study)


def rational(value):
    value = Fraction(value)
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))


def decode(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


def interval(bounds):
    lower, upper = map(Fraction, bounds)
    return dict(lower=rational(lower), upper=rational(upper), width=rational(upper-lower))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')


def teacher(index):
    times = [j/3 for j in range(10)]
    rows = [[float(100*index+j)]+[float(k/64) for k in range(1, 56)] for j in range(10)]
    model = dict(spots=[100.]*5, strike=100., rate=.05, dividends=[.1]*5,
        volatilities=[.2]*5, exercise_times=times,
        correlation=[[float(i == j) for j in range(5)] for i in range(5)])
    signatures = [()]+[signature for degree in range(1, 4) for signature in combinations_with_replacement(range(5), degree)]
    powers = [[signature.count(i) for i in range(5)] for signature in signatures]
    # These are lineage labels only. No generator is initialised or called.
    inputs = dict(model=model, design=dict(root_seed=2026100410+index, value_paths=50000))
    polynomial = dict(root_seed=2026100410+index, sample_count=256, value_fitting_paths=50000,
        value_spawn_key=[1], degrees={'3': dict(coefficients=rows, basis_exponents=powers)})
    return inputs, polynomial


def literal_average(coefficients):
    signatures = [()]+[signature for degree in range(1, 4) for signature in combinations_with_replacement(range(5), degree)]
    totals = [Fraction(0)]*7; counts = [0]*7
    for signature, coefficient in zip(signatures, coefficients):
        if not signature:
            orbit = 0
        elif len(signature) == 1:
            orbit = 1
        elif len(signature) == 2:
            orbit = 2 if signature[0] == signature[1] else 3
        else:
            orbit = {1: 4, 2: 5, 3: 6}[len(set(signature))]
        totals[orbit] += Fraction(coefficient); counts[orbit] += 1
    return tuple(value/count for value, count in zip(totals, counts))


def fixture_tree(root):
    retained = root/'data/private/retained'
    for index in range(8):
        inputs, polynomial = teacher(index)
        write_json(retained/f'fit_{index:02d}/inputs.json', inputs)
        write_json(retained/f'fit_{index:02d}/polynomial_view.json', polynomial)
        write_json(retained/f'fit_{index:02d}/assessment_attempt.json',
            dict(input_sha256=study.sha(retained/f'fit_{index:02d}/inputs.json'),
                polynomial_view_sha256=study.sha(retained/f'fit_{index:02d}/polynomial_view.json')))
    original_name = 'src/retained_original_component.py'
    original = root/original_name
    original.parent.mkdir(parents=True, exist_ok=True); original.write_text('retained numerical source\n')
    write_json(root/'data/private/revision/2026-10-05/financial_decision_v1/study/design.json',
        dict(source_sha256={original_name: hashlib.sha256(original.read_bytes()).hexdigest()},
            fits=[dict(index=i, root_seed=2026100410+i) for i in range(8)],
            retained_input_sha256={f'fit_{i:02d}/inputs.json': study.sha(retained/f'fit_{i:02d}/inputs.json') for i in range(8)}))
    for name in study.SOURCE_NAMES:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('new study source fixture\n')
    return retained, original


def fixture_reference(root):
    path = root/'data/private/revision/2026-10-05/financial_decision_v1/study/design.json'
    return patch.multiple(study, REFERENCE_DESIGN_PATH=path, REFERENCE_DESIGN_SHA=study.sha(path))


def signatures(dimension=5, degree=3):
    return ((),)+tuple(signature for order in range(1, degree+1) for signature in combinations_with_replacement(range(dimension), order))


def preparation(snapshot):
    cases = []
    for index in range(8):
        inputs, polynomial = teacher(index)
        model = inputs['model']
        for case_name in ('exchangeable_reference', 'published_asymmetric_volatility'):
            vols = tuple(Fraction(x) for x in model['volatilities']) if case_name == 'exchangeable_reference' else tuple(Fraction(2, 25)+Fraction(8, 25)*i/4 for i in range(5))
            law = dict(rate=rational(model['rate']), dividends=[rational(x) for x in model['dividends']],
                volatilities=[rational(x) for x in vols], correlation=rational(0))
            dates = []
            for j in range(1, 9):
                average = literal_average(polynomial['degrees']['3']['coefficients'][j])
                selected = list(average); selected[1] += Fraction(1, 4)
                dates.append(dict(date_index=j, previous_time=rational(model['exercise_times'][j-1]),
                current_time=rational(model['exercise_times'][j]),
                retained_coefficients=[rational(x) for x in polynomial['degrees']['3']['coefficients'][j]],
                averaging_coefficients=[rational(x) for x in average],
                nominal_projection=dict(frozen_orbit_coefficients=[rational(x) for x in selected],
                    diagnostics=dict(exact_optimiser_certified=False, degree=3, monomial_count=56,
                        previous_time=rational(model['exercise_times'][j-1]), current_time=rational(model['exercise_times'][j]),
                        model=dict(dimension=5, initial_scaled_state=[rational(1)]*5,
                            family='common_volatility_exchangeable' if case_name == 'exchangeable_reference' else 'independent_heterogeneous_volatility', **law)))))
            cases.append(dict(fit_index=index, model_case=case_name,
                initial_scaled_state=[rational(1)]*5, model=law, dates=dates))
    return dict(author='Amanjeet Singh', completed=True, cases=cases,
        source_snapshot_before=deepcopy(snapshot), source_snapshot_after=deepcopy(snapshot),
        models=2, teachers=8, nonterminal_dates=8, terminal_payoff_retained=True,
        new_sampling_executed=False, new_policy_fitting_executed=False)


def fake_core(old=(2, 3), new=(1, Fraction(3, 2)), *, on_increment=None, force_exchangeable_strict=False):
    def energy(c, b, initial, t0, t1, **kwargs):
        if on_increment is not None:
            on_increment()
        return dict(candidate='nominal_projection' if b[1] > Fraction(1, 4) else 'averaging',
            coefficients=[rational(x) for x in c], initial=[rational(x) for x in initial],
            t0=rational(t0), t1=rational(t1), precision=kwargs['precision'],
            exchangeable=len(set(kwargs['volatilities'])) == 1)
    def combined(records, **kwargs):
        candidate = records[0]['candidate']
        bounds = old if candidate == 'averaging' or (records[0]['exchangeable'] and not force_exchangeable_strict) else new
        return dict(terminal_energy_certificate=interval(bounds),
            absolute_expected_dual_gap_difference_upper=rational(7 if candidate == 'averaging' else 6))
    return SimpleNamespace(rational_record=rational, decode_rational=decode,
        complete_monomial_indices=signatures, project_coefficients=Mock(side_effect=lambda c, d, degree: literal_average(c)),
        decode_interval=lambda record: (decode(record['lower']), decode(record['upper'])),
        certify_residual_energy=Mock(side_effect=energy), combine_energy_certificates=Mock(side_effect=combined))


class SnapshotChecks(unittest.TestCase):
    def test_all_sixteen_retained_files_and_registered_sources_are_bound(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained, original = fixture_tree(root)
            with fixture_reference(root):
                first = study.snapshot(root, retained)
                self.assertEqual(len(first['retained_input_sha256']), 16)
                self.assertEqual(set(first['source_sha256']), set(study.SOURCE_NAMES))
                self.assertIn('src/retained_original_component.py', first['original_registered_source_sha256'])
                changed = retained/'fit_07/polynomial_view.json'
                initial_bytes = changed.read_bytes()
                changed.write_bytes(initial_bytes+b'\n')
                with self.assertRaises(ValueError):
                    study.snapshot(root, retained)
                changed.write_bytes(initial_bytes)
                original.write_text('altered original numerical source\n')
                with self.assertRaises(ValueError):
                    study.snapshot(root, retained)

    def test_reference_design_and_teacher_cohort_are_sealed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained, _ = fixture_tree(root)
            path = root/'data/private/revision/2026-10-05/financial_decision_v1/study/design.json'
            with fixture_reference(root):
                original = path.read_text(); path.write_text(original+'\n')
                with self.assertRaises(ValueError):
                    study.snapshot(root, retained)
            value = json.loads(original); value['fits'].pop(); write_json(path, value)
            with fixture_reference(root):
                with self.assertRaises(ValueError):
                    study.snapshot(root, retained)


class PreparationChecks(unittest.TestCase):
    def mocked_core(self):
        core = fake_core()
        def fitted(c, initial, t0, t1, **kwargs):
            selected = list(literal_average(c)); selected[1] += Fraction(1, 4)
            frozen = dict(frozen_orbit_coefficients=[rational(x) for x in selected],
                diagnostics=dict(exact_optimiser_certified=False))
            return SimpleNamespace(record=lambda: frozen)
        core.fit_energy_projection = Mock(side_effect=fitted)
        return core

    def test_preparation_retains_all_cases_dates_models_and_source_coefficients(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained, _ = fixture_tree(root)
            with fixture_reference(root):
                bound = study.snapshot(root, retained)
            output = root/'data/private/projection'; output.mkdir()
            core = self.mocked_core()
            with patch.object(study, 'snapshot', return_value=bound), patch.object(study, 'print', create=True):
                result = study.prepare(root, retained, output, core)
            self.assertEqual(core.fit_energy_projection.call_count, 128)
            self.assertEqual([(row['fit_index'], row['model_case']) for row in result['cases']],
                [(i, case) for i in range(8) for case in study.MODEL_CASES])
            self.assertEqual((result['teachers'], result['models'], result['nonterminal_dates']), (8, 2, 8))
            self.assertIs(result['new_sampling_executed'], False)
            self.assertIs(result['new_policy_fitting_executed'], False)
            self.assertTrue(result['terminal_payoff_retained'])
            self.assertTrue(study.validate_prepared(root, retained, result, core))
            for call, row_index in zip(core.fit_energy_projection.call_args_list, range(128)):
                args, kwargs = call
                fit_index = row_index//16; date = row_index%8+1
                inputs, polynomial = teacher(fit_index)
                self.assertEqual(args[0], polynomial['degrees']['3']['coefficients'][date])
                self.assertEqual(args[1], (Fraction(1),)*5)
                self.assertEqual(args[2:4], tuple(Fraction(x) for x in inputs['model']['exercise_times'][date-1:date+1]))
                self.assertEqual(kwargs['correlation'], 0)
            self.assertEqual(study.read(output/'prepared.json'), result)

    def test_changed_teacher_basis_root_calendar_and_normalisation_are_rejected(self):
        changes = [('inputs.json', lambda record: record['model'].__setitem__('strike', 90.)),
            ('inputs.json', lambda record: record['design'].__setitem__('root_seed', 2026999912)),
            ('inputs.json', lambda record: record['model']['exercise_times'].__setitem__(2, record['model']['exercise_times'][1])),
            ('polynomial_view.json', lambda record: record['degrees']['3']['basis_exponents'].__setitem__(1, [0]*5)),
            ('polynomial_view.json', lambda record: record['degrees']['3']['coefficients'].pop())]
        for filename, change in changes:
            with self.subTest(filename=filename), TemporaryDirectory() as temporary:
                root = Path(temporary); retained, _ = fixture_tree(root)
                path = retained/'fit_00'/filename; value = study.read(path); change(value); write_json(path, value)
                with self.assertRaises(ValueError):
                    study.teacher(root, retained, 0, self.mocked_core())

    def test_source_change_during_projection_prevents_a_completed_preparation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained, _ = fixture_tree(root)
            with fixture_reference(root):
                bound = study.snapshot(root, retained)
            output = root/'data/private/changed_projection'; output.mkdir()
            after = deepcopy(bound); after['source_sha256']['changed'] = 'later source'
            with patch.object(study, 'snapshot', side_effect=[bound, after]), patch.object(study, 'print', create=True):
                with self.assertRaises(ValueError):
                    study.prepare(root, retained, output, self.mocked_core())
            self.assertTrue((output/'attempt.json').is_file())
            self.assertFalse((output/'prepared.json').exists())


class CertificationChecks(unittest.TestCase):
    def run_case(self, old=(2, 3), new=(1, Fraction(3, 2)), *, mutate=None, snapshots=None, callback=None,
            after_seal=None, force_exchangeable_strict=False):
        temporary = TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name); retained, _ = fixture_tree(root)
        with fixture_reference(root):
            bound = study.snapshot(root, retained)
        prep = root/'data/private/preparation'; output = root/'data/private/certification'; output.mkdir()
        prepared = preparation(bound)
        if mutate is not None:
            mutate(prepared)
        write_json(prep/'prepared.json', prepared)
        sealed = study.sha(prep/'prepared.json')
        if after_seal is not None:
            after_seal(prepared); write_json(prep/'prepared.json', prepared)
        core = fake_core(old, new, on_increment=None if callback is None else lambda: callback(prep),
            force_exchangeable_strict=force_exchangeable_strict)
        context = patch.object(study, 'snapshot', side_effect=snapshots) if snapshots is not None else patch.object(study, 'snapshot', return_value=bound)
        with context, patch.object(study, 'print', create=True):
            result = study.certify(root, retained, prep, output, core, sealed)
        return result, core, prep, output, prepared

    def test_full_eight_by_two_by_eight_counts_preserve_time_and_coefficient_identity(self):
        result, core, prep, output, prepared = self.run_case()
        self.assertEqual(len(result['cases']), 16)
        self.assertEqual(core.certify_residual_energy.call_count, 256)
        self.assertEqual(core.combine_energy_certificates.call_count, 32)
        self.assertEqual((result['teachers'], result['model_cases'], result['increment_certificates']), (8, 2, 256))
        calls = core.certify_residual_energy.call_args_list
        position = 0
        for row in prepared['cases']:
            for date in row['dates']:
                for candidate in ('averaging', 'nominal_projection'):
                    args, kwargs = calls[position]; position += 1
                    self.assertEqual(args[0], tuple(map(decode, date['retained_coefficients'])))
                    self.assertEqual(args[2], (Fraction(1),)*5)
                    self.assertEqual(args[3:5], (decode(date['previous_time']), decode(date['current_time'])))
                    self.assertEqual(kwargs['precision'], 128)
        self.assertEqual(study.read(output/'result.json')['prepared_sha256'], study.sha(prep/'prepared.json'))
        for flag in ('new_sampling_executed', 'new_policy_fitting_executed', 'absolute_pricing_error_established',
                'market_validation_established', 'substantive_original_research_contribution_established'):
            self.assertIs(result[flag], False)
        self.assertTrue(result['terminal_energy_contribution_exact_zero'])
        self.assertIn('No exact numerical optimality is claimed', result['cases'][0]['selection_scope'])

    def test_strict_distinct_intervals_select_only_the_asymmetric_case(self):
        result, _, _, _, _ = self.run_case(old=(2, 3), new=(1, Fraction(3, 2)))
        for row in result['cases']:
            self.assertEqual(row['strict_energy_reduction_certified'], row['model_case'] != 'exchangeable_reference')
            expected = 'averaging' if row['model_case'] == 'exchangeable_reference' else 'nominal_projection'
            self.assertEqual(row['selected_candidate'], expected)
            self.assertEqual(decode(row['reference_hedge_rms_difference_upper']), 7 if expected == 'averaging' else 6)

    def test_a_strict_exchangeable_reduction_is_rejected_as_a_contradiction(self):
        with self.assertRaises(ArithmeticError):
            self.run_case(force_exchangeable_strict=True)

    def test_touching_overlapping_and_worse_intervals_keep_averaging(self):
        for old, new in [((2, 3), (1, 2)), ((2, 3), (Fraction(5, 2), Fraction(7, 2))), ((2, 3), (4, 5))]:
            with self.subTest(old=old, new=new):
                result, _, _, _, _ = self.run_case(old=old, new=new)
                self.assertTrue(all(not row['strict_energy_reduction_certified'] and row['selected_candidate'] == 'averaging' for row in result['cases']))

    def test_initial_preparation_binding_change_fails_before_numerical_calls(self):
        with self.assertRaises(ValueError):
            self.run_case(mutate=lambda record: record['source_snapshot_after'].update(source_sha256={'changed': 'input'}))

    def test_prestart_preparation_changes_and_invalid_cohorts_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(after_seal=lambda record: record['cases'].pop())
        changes = [lambda record: record['cases'].pop(),
            lambda record: record['cases'].__setitem__(1, deepcopy(record['cases'][0])),
            lambda record: record['cases'][0]['dates'].pop(),
            lambda record: record['cases'][0]['dates'].reverse(),
            lambda record: record['cases'][0]['dates'][0]['retained_coefficients'].__setitem__(1, rational(999)),
            lambda record: record['cases'][0]['dates'][0].__setitem__('current_time', rational(1)),
            lambda record: record['cases'][0]['dates'][0]['averaging_coefficients'].__setitem__(1, rational(999))]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.run_case(mutate=change)

    def test_source_change_and_preparation_change_during_certification_fail(self):
        with self.assertRaises(ValueError):
            self.run_case(snapshots=[{'bound': 1}, {'bound': 2}],
                mutate=lambda record: record.update(source_snapshot_after={'bound': 1}))
        def change_once(prep):
            path = prep/'prepared.json'
            if not path.read_text().endswith('\n\n'):
                path.write_text(path.read_text()+'\n')
        with self.assertRaises(ValueError):
            self.run_case(callback=change_once)


class PrivateExecutionChecks(unittest.TestCase):
    def invoke(self, root, retained, output, stage='prepare', preparation_path=None, *, error=None):
        argv = ['study', stage, '--root', str(root), '--retained-study', str(retained), '--output', str(output)]
        if preparation_path is not None:
            argv += ['--preparation', str(preparation_path), '--preparation-sha256', '0'*64]
        load = Mock(side_effect=error) if error is not None else Mock(return_value=object())
        with (patch.object(sys, 'argv', argv), patch.object(study, 'load', load),
                patch.object(study, 'prepare') as prepare_call, patch.object(study, 'certify') as certify_call,
                patch.object(study.time, 'process_time', side_effect=[10., 12.]),
                patch.object(study.time, 'perf_counter', side_effect=[20., 23.])):
            study.main()
        return load, prepare_call, certify_call

    def test_success_cost_record_and_failure_record_are_private_and_explicit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained = root/'data/private/retained'; retained.mkdir(parents=True)
            output = root/'data/private/preparation'
            _, prepare_call, _ = self.invoke(root, retained, output)
            prepare_call.assert_called_once()
            record = study.read(output/'execution.json')
            self.assertEqual((record['cpu_seconds'], record['wall_seconds']), (2., 3.))
            self.assertIs(record['completed'], True)
            self.assertIn('process startup', record['scope'])
            failed = root/'data/private/failed'
            with self.assertRaises(RuntimeError):
                self.invoke(root, retained, failed, error=RuntimeError('fixture failure'))
            record = study.read(failed/'failure.json')
            self.assertFalse(record['completed'])
            self.assertEqual(record['message'], 'fixture failure')
            self.assertEqual((record['cpu_seconds'], record['wall_seconds']), (2., 3.))

    def test_existing_public_and_teacher_overlapping_outputs_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained = root/'data/private/retained'; retained.mkdir(parents=True)
            existing = root/'data/private/existing'; existing.mkdir(); sentinel = existing/'sentinel'; sentinel.write_text('keep')
            for output in (existing, root/'public/result', retained/'new_output'):
                with self.subTest(output=output):
                    with self.assertRaises(ValueError):
                        self.invoke(root, retained, output)
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertFalse((root/'public/result').exists())
            self.assertFalse((retained/'new_output').exists())
            outside = root/'public/symlink_target'; outside.mkdir(parents=True)
            link = root/'data/private/symlink_output'; link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.invoke(root, retained, link/'new_output')
            self.assertFalse((outside/'new_output').exists())

    def test_certification_requires_private_separate_preparation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); retained = root/'data/private/retained'; retained.mkdir(parents=True)
            with self.assertRaises(ValueError):
                self.invoke(root, retained, root/'data/private/missing_preparation', stage='certify')
            with self.assertRaises(ValueError):
                self.invoke(root, retained, root/'data/private/public_preparation', stage='certify', preparation_path=root/'public/prepared')
            prep = root/'data/private/frozen_preparation'; prep.mkdir()
            nested = prep/'new_output'
            with self.assertRaises(ValueError):
                self.invoke(root, retained, nested, stage='certify', preparation_path=prep)
            self.assertFalse(nested.exists())


if __name__ == '__main__':
    unittest.main()
