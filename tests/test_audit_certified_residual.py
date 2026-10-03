"""Exact, stipulated certificate fixtures; no empirical observations or paths."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_certified_residual import check_model


def fixture():
    left = dict(knots=[0., 1., 2.], values=[2., 1., 0.], tail_slope=0.)
    right = dict(knots=[0., 2.], values=[0., 0.], tail_slope=0.)
    spec = dict(case=0, approximation_fingerprint='stipulated mathematical fixture',
        model=dict(volatility=.25, premium_factor=.5, dividends=[]),
        option=dict(strike=2., maturity=1., kind='put'),
        nodes=[dict(time=0., before=None, after=left), dict(time=1., before=None, after=right)],
        decisions=[dict(time=0., phase='after', settlement_factor=1.),
                   dict(time=1., phase='after', settlement_factor=1.)])
    points = [dict(state=str(x), value=str(2-x), mean_lower='0', mean_upper='0',
                   residual_lower=str(2-x), residual_upper=str(2-x)) for x in range(3)]
    cells = [dict(a=str(x), b=str(x+1), lower=str(1-x), upper=str(2-x),
                  curvature_upper='0', chord_allowance='0') for x in range(2)]
    edge = dict(kind='diffusion', from_time='0', to_time='1', lower='0', upper='2',
        candidate_inf_upper='0', candidate_sup_lower='2', tolerance='1/1000',
        coverage_complete=True, tolerance_met=True, zero_volatility=False, rounds=0,
        sd_enclosure=['1/4', '1/4'], points=points, cells=cells,
        tail=dict(start='2', lower='0', upper='0', limit='0'))
    rights = [dict(vertex=0, time='0', phase='after', threshold='2',
                   knots=['0', '1', '2'], values=['0', '0', '0'], lower='0', upper='0', tail='0'),
              dict(vertex=1, time='1', phase='after', threshold='2',
                   knots=['0', '2'], values=['2', '0'], lower='0', upper='2', tail='0')]
    result = dict(case=0, approximation_fingerprint=spec['approximation_fingerprint'],
        gain_bound='4', edges=[edge], rights=rights,
        pairs=[dict(earlier=0, later=1, lower='-2', upper='2')],
        coverage_complete=True, tolerance_met=True, ideal_model_bound_certified=True,
        recorded_floating_point_gains_certified=False)
    return spec, result


class WitnessAuditTests(unittest.TestCase):
    def test_valid_exact_fixture(self):
        spec, result = fixture()
        checked = check_model(spec, result, Fraction(1, 1000))
        self.assertEqual(checked['gain_bound'], '4')
        self.assertEqual((checked['cells'], checked['points'], checked['pairs']), (2, 3, 1))

    def test_mutated_cell_endpoint_rejected(self):
        spec, result = fixture()
        result['edges'][0]['cells'][0]['b'] = '3/2'
        with self.assertRaisesRegex(ValueError, 'arithmetic differs: b'):
            check_model(spec, result, Fraction(1, 1000))

    def test_omitted_cell_rejected(self):
        spec, result = fixture()
        result['edges'][0]['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'exactly one cell'):
            check_model(spec, result, Fraction(1, 1000))

    def test_omitted_left_knot_rejected_even_with_contiguous_cells(self):
        spec, result = fixture()
        edge = result['edges'][0]
        edge['points'].pop(1)
        edge['cells'] = [dict(a='0', b='2', lower='0', upper='2',
                              curvature_upper='0', chord_allowance='0')]
        with self.assertRaisesRegex(ValueError, 'missing knots'):
            check_model(spec, result, Fraction(1, 1000))

    def test_changed_pair_bound_rejected(self):
        spec, result = fixture()
        result['pairs'][0]['lower'] = '-1'
        with self.assertRaisesRegex(ValueError, 'arithmetic differs: lower'):
            check_model(spec, result, Fraction(1, 1000))

    def test_invalid_exact_volatility_enclosure_rejected(self):
        spec, result = fixture()
        result['edges'][0]['sd_enclosure'] = ['1/8', '1/8']
        with self.assertRaisesRegex(ValueError, 'exact model value'):
            check_model(spec, result, Fraction(1, 1000))

    def test_wrong_tail_limit_rejected(self):
        spec, result = fixture()
        result['edges'][0]['tail']['limit'] = '1'
        with self.assertRaisesRegex(ValueError, 'arithmetic differs: limit'):
            check_model(spec, result, Fraction(1, 1000))

    def test_invalid_payoff_defect_rejected(self):
        spec, result = fixture()
        result['rights'][1]['values'][0] = '1'
        with self.assertRaisesRegex(ValueError, 'Payoff defect'):
            check_model(spec, result, Fraction(1, 1000))


if __name__ == '__main__':
    unittest.main()
