"""Run the frozen prospective audit with a JSON-safe console report.

The registered auditor is preserved byte for byte. This entry point serialises
its exact rational return values and binds itself in the new audit record.
"""

import argparse
import json
from pathlib import Path

import audit_prospective_allocation as core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=core.BASE/'registration_v1')
    parser.add_argument('--run', type=Path, default=core.BASE/'run_v1')
    parser.add_argument('--output', type=Path, default=core.BASE/'audit_v2')
    args = parser.parse_args()
    result = core.audit(args.registration, args.run, args.output)
    source = Path(__file__).resolve()
    result['evidence_sha256'][str(source.relative_to(core.ROOT))] = core.sha(source)
    core.save(args.output/'checks.json', result)
    summary = {k: result[k] for k in ('complete', 'cases', 'arms', 'primary_pairs',
        'warmup_pairs', 'contrasts', 'critical_value', 'finite_sample_coverage_certified')}
    summary['verified_evidence_files'] = len(result['evidence_sha256'])
    print(json.dumps(core.encode(summary), allow_nan=False))


if __name__ == '__main__':
    main()
