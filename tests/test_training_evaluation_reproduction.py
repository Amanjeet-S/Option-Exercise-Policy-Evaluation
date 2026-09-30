"""Storage representation must not change the numerical reproduction check."""

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from reproduce_training_evaluation import canonical_numerical_record


def test_json_round_trip_retains_exact_numerical_record():
    block = dict(training=[dict(policy=dict(coefficients=(1.25, -.5),
                 decisions=({'time': .25}, {'time': 1.})), elapsed_seconds=2.)],
                 gain=dict(value=.03125, standard_error=.001), phases={'build': 1.})
    saved = json.loads(json.dumps(block))
    saved['training'][0]['elapsed_seconds'] = 99.
    assert canonical_numerical_record(block) == canonical_numerical_record(saved)
    altered = copy.deepcopy(saved)
    altered['training'][0]['policy']['coefficients'][0] += 1e-12
    assert canonical_numerical_record(block) != canonical_numerical_record(altered)


def test_non_finite_numerical_record_is_rejected():
    with pytest.raises(ValueError):
        canonical_numerical_record({'value': float('nan')})
