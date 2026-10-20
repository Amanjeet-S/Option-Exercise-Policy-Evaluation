"""Build the reviewed source inventory for the current public-model revision.

The default command checks the selection and prints the proposed manifest.
Only --write creates or replaces replication/current_source_manifest.json.
No numerical study, provider request or publication action is performed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'replication/current_source_manifest.json'

# Each path is a reviewed source selection. New files require an explicit edit.
REVIEWED_GROUPS = {
    'reader_documentation': (
        'README.md', 'SOURCES.md', 'PROVENANCE.md', 'THIRD_PARTY_NOTICES.md',
        'LICENSE', 'LICENSING.md', 'CONTRIBUTING.md', '.gitignore', 'completion.json',
        'cpp/README.md', 'data/README.md', 'docs/development.md',
        'docs/publication_policy.md', 'docs/reproducing_paper.md',
        'notebooks/README.md', 'paper/README.md', 'paper/contribution_review.md',
        'replication/README.md', 'scripts/README.md',
    ),
    'environment': (
        'pyproject.toml', 'requirements.lock', 'requirements-notebooks.lock',
        'requirements-certification.lock', '.github/workflows/ci.yml',
    ),
    'paper_source': (
        'paper/main.tex', 'paper/historical_results.tex', 'paper/references.bib',
        'paper/revision_results.json', 'paper/Makefile',
    ),
    'verification': (
        'replication/verify_sources.py', 'paper/verification/revision_results.py',
        'paper/verification/test_revision_results.py',
        'paper/verification/allocation_probability.py',
        'paper/verification/conditional_drift.py',
        'paper/verification/test_allocation_probability.py',
        'paper/verification/test_conditional_drift.py',
    ),
    'notebook': (
        'notebooks/pricing_and_stopping.ipynb',
        'notebooks/numerical_validation.ipynb',
        'notebooks/conditional_research_replication.ipynb',
    ),
    'native_source': ('cpp/bermudan_tree.cpp',),
    'numerical_source': (
        'src/qf_research/__init__.py',
        'src/qf_research/computational/__init__.py',
        'src/qf_research/computational/funded_kernel.py',
        'src/qf_research/market_data/__init__.py',
        'src/qf_research/market_data/calendar_robustness.py',
        'src/qf_research/market_data/dense_quotes.py',
        'src/qf_research/market_data/event_calendar.py',
        'src/qf_research/market_data/front_funding.py',
        'src/qf_research/market_data/funded_inputs.py',
        'src/qf_research/market_data/occ_series.py',
        'src/qf_research/market_data/option_quotes.py',
        'src/qf_research/market_data/prospective_inputs.py',
        'src/qf_research/market_data/prospective_sample.py',
        'src/qf_research/market_data/quote_review.py',
        'src/qf_research/market_data/settlement_funding.py',
        'src/qf_research/market_data/short_ois.py',
        'src/qf_research/market_data/valuation_inputs.py',
        'src/qf_research/optimal_stopping/__init__.py',
        'src/qf_research/optimal_stopping/adaptive_martingale_compression.py',
        'src/qf_research/optimal_stopping/allocation_sensitivity.py',
        'src/qf_research/optimal_stopping/allocation_variance_sensitivity.py',
        'src/qf_research/optimal_stopping/allocation_workload.py',
        'src/qf_research/optimal_stopping/analytical.py',
        'src/qf_research/optimal_stopping/cash_dividends.py',
        'src/qf_research/optimal_stopping/component_confidence.py',
        'src/qf_research/optimal_stopping/component_ratio.py',
        'src/qf_research/optimal_stopping/contracts.py',
        'src/qf_research/optimal_stopping/cost_budget.py',
        'src/qf_research/optimal_stopping/dividend_sensitivity.py',
        'src/qf_research/optimal_stopping/escrowed_dividends.py',
        'src/qf_research/optimal_stopping/exercise_dominance.py',
        'src/qf_research/optimal_stopping/exercise_redundancy.py',
        'src/qf_research/optimal_stopping/experiments.py',
        'src/qf_research/optimal_stopping/expiry_extension.py',
        'src/qf_research/optimal_stopping/finite_difference.py',
        'src/qf_research/optimal_stopping/funded_cash.py',
        'src/qf_research/optimal_stopping/funded_dual.py',
        'src/qf_research/optimal_stopping/funded_policy.py',
        'src/qf_research/optimal_stopping/joint_input_expiry.py',
        'src/qf_research/optimal_stopping/least_squares.py',
        'src/qf_research/optimal_stopping/martingale_compression.py',
        'src/qf_research/optimal_stopping/monetary_two_asset.py',
        'src/qf_research/optimal_stopping/multivariate_benchmark.py',
        'src/qf_research/optimal_stopping/penalty_intervals.py',
        'src/qf_research/optimal_stopping/policy_gain.py',
        'src/qf_research/optimal_stopping/policy_transfer.py',
        'src/qf_research/optimal_stopping/positive_martingale_energy.py',
        'src/qf_research/optimal_stopping/prospective_allocation.py',
        'src/qf_research/optimal_stopping/prospective_evaluation.py',
        'src/qf_research/optimal_stopping/reference_policy.py',
        'src/qf_research/optimal_stopping/residual_envelope.py',
        'src/qf_research/optimal_stopping/sequential_inference.py',
        'src/qf_research/optimal_stopping/settlement.py',
        'src/qf_research/optimal_stopping/simulation.py',
        'src/qf_research/optimal_stopping/sparse_pricing_kernel.py',
        'src/qf_research/optimal_stopping/symmetry.py',
        'src/qf_research/optimal_stopping/terminal_repair.py',
        'src/qf_research/optimal_stopping/terminal_training.py',
        'src/qf_research/optimal_stopping/training_allocation.py',
        'src/qf_research/optimal_stopping/training_evaluation.py',
        'src/qf_research/optimal_stopping/tree.py',
        'src/qf_research/optimal_stopping/value_martingale.py',
    ),
    'research_script': tuple('scripts/'+name+'.py' for name in (
        'audit_research_revision', 'build_current_inventory',
        'certify_european_gain_tail', 'certify_financial_decision',
        'certify_gain_replay', 'certify_residual_envelope',
        'certify_max_hinge', 'certify_multivariate_gap',
        'certify_policy_loss_reference', 'certify_polynomial_dual',
        'certify_restricted_gain', 'correct_financial_decision_inference',
        'execute_release_notebooks', 'plot_research_revision',
        'prepare_financial_decision', 'prepare_monetary_two_asset',
        'refine_european_gain', 'refine_multivariate_gap',
        'reproduce_current_study', 'reproduce_paper',
        'study_european_gain_allocation',
        'study_financial_decision', 'study_frozen_gap_extension',
        'study_martingale_compression', 'study_multivariate_gap_refinement',
        'study_multivariate_matrix', 'study_policy_certification',
        'study_replicated_gap_refinement', 'study_uniform_gap_allocation',
        'validate_clipped_gap_coverage', 'validate_gap_coverage',
        'validate_multivariate_benchmark', 'validate_polynomial_dual',
        'validate_reference_policy', 'validate_funded_dual', 'verify_bermudan_tree',
        'test_certified_residual',
    )),
    'test_source': tuple('tests/'+name+'.py' for name in (
        'test_adaptive_martingale_compression', 'test_audit_research_revision',
        'test_certify_european_gain_tail', 'test_certify_financial_decision',
        'test_certify_max_hinge', 'test_certify_multivariate_gap',
        'test_certify_policy_loss_reference', 'test_certify_polynomial_dual',
        'test_certify_restricted_gain', 'test_certify_gain_replay',
        'test_clipped_gap_coverage',
        'test_correct_financial_decision_inference', 'test_current_inventory',
        'test_frozen_gap_extension', 'test_gap_coverage',
        'test_martingale_compression', 'test_monetary_two_asset',
        'test_multivariate_benchmark', 'test_notebook_execution_metadata',
        'test_optimal_stopping', 'test_penalty_intervals',
        'test_policy_certification', 'test_policy_gain', 'test_polynomial_dual',
        'test_positive_martingale_energy', 'test_prepare_financial_decision',
        'test_reference_policy', 'test_refine_european_gain',
        'test_refine_multivariate_gap', 'test_replicated_gap_refinement',
        'test_replication_sources', 'test_reproduce_current_study',
        'test_reproduce_paper', 'test_sequential_inference',
        'test_sparse_pricing_kernel', 'test_study_european_gain_allocation',
        'test_study_financial_decision', 'test_study_martingale_compression',
        'test_symmetry', 'test_uniform_gap_allocation',
    )),
}

DOCUMENTATION_ROOT_FILES = {
    'README.md', 'SOURCES.md', 'PROVENANCE.md', 'THIRD_PARTY_NOTICES.md',
    'LICENSE', 'LICENSING.md', 'CONTRIBUTING.md', '.gitignore', 'completion.json',
}


def load_verifier(root):
    """Load the existing standalone verifier from the selected checkout."""
    path = root/'replication/verify_sources.py'
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError('The standalone verifier must be a regular local source')
    spec = importlib.util.spec_from_file_location('current_inventory_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reviewed_selection(groups, verifier):
    """Reject aliases, duplicate selections and files outside the review scope."""
    if not isinstance(groups, dict) or not groups:
        raise ValueError('A nonempty reviewed source selection is required')
    selected = {}
    for category, paths in groups.items():
        if category not in REVIEWED_GROUPS or not isinstance(paths, (tuple, list)) or not paths:
            raise ValueError('Unknown or empty reviewed source category')
        for name in paths:
            path = verifier.relative_path(name)
            if name == MANIFEST or name == 'replication/source_manifest.json':
                raise ValueError('Inventory manifests are outside the source selection')
            if name in selected:
                raise ValueError('Duplicate reviewed source path: '+name)
            parts = path.parts
            allowed = (
                category == 'reader_documentation' and
                (name in DOCUMENTATION_ROOT_FILES or
                 (len(parts) == 2 and parts[0] in {'cpp','docs','notebooks','paper','replication','scripts'}
                  and path.suffix == '.md') or name == 'data/README.md') or
                category == 'environment' and
                (name in {'pyproject.toml','requirements.lock','requirements-notebooks.lock',
                          'requirements-certification.lock','.github/workflows/ci.yml'}) or
                category == 'paper_source' and len(parts) == 2 and parts[0] == 'paper'
                and path.suffix in {'.tex','.bib','.json'} or
                category == 'paper_source' and name == 'paper/Makefile' or
                category == 'verification' and
                (name == 'replication/verify_sources.py' or
                 (len(parts) == 3 and parts[:2] == ('paper','verification') and path.suffix == '.py')) or
                category == 'notebook' and len(parts) == 2 and parts[0] == 'notebooks' and path.suffix == '.ipynb' or
                category == 'native_source' and len(parts) == 2 and parts[0] == 'cpp' and path.suffix == '.cpp' or
                category == 'numerical_source' and parts[:2] == ('src','qf_research') and path.suffix == '.py' or
                category == 'research_script' and len(parts) == 2 and parts[0] == 'scripts' and path.suffix == '.py' or
                category == 'test_source' and len(parts) == 2 and parts[0] == 'tests' and path.suffix == '.py'
            )
            forbidden = {'.git','.venv','.venv-certification','__pycache__','private','raw'}
            if not allowed or forbidden.intersection(parts):
                raise ValueError('File lies outside the reviewed public source scope: '+name)
            selected[name] = category
    return dict(sorted(selected.items()))


def source_units(name, data):
    """Read Python syntax without executing research or notebook code."""
    if name.endswith('.py'):
        return [(name, ast.parse(data.decode('utf-8'), filename=name))]
    if name.endswith('.ipynb'):
        document = json.loads(data)
        return [(name, ast.parse(''.join(cell['source']), filename=name+':'+str(i)))
                for i, cell in enumerate(document['cells']) if cell['cell_type'] == 'code']
    return []


def external_imports(blobs, *, root=None):
    """Record static third-party imports and check ordinary package imports.

    Dynamic imports, generated names and compiler/runtime dependencies require
    the separate execution and environment checks. This is not a closure proof.
    """
    local_modules = {}
    for name in blobs:
        if name.endswith('.py'):
            stem = name[:-3].replace('/', '.')
            if stem.startswith('src.'):
                stem = stem[4:]
            if stem.endswith('.__init__'):
                stem = stem[:-9]
            local_modules[stem] = name
    local_top = {name.split('.')[0] for name in local_modules}
    local_base = {Path(name).stem for name in blobs if name.endswith('.py')}
    imports = {}
    for name, data in blobs.items():
        for _, tree in source_units(name, data):
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        if not name.startswith('src/'):
                            raise ValueError('Relative import outside the declared package: '+name)
                        parent = name[4:-3].replace('/','.').split('.')[:-node.level]
                        prefix = '.'.join(parent+([node.module] if node.module else []))
                        modules = [prefix]
                        if not node.module:
                            modules += [prefix+'.'+alias.name for alias in node.names]
                    elif node.module:
                        modules = [node.module]
                        if node.module == 'qf_research' or node.module in local_modules:
                            modules += [node.module+'.'+alias.name for alias in node.names
                                        if alias.name != '*'
                                        and node.module+'.'+alias.name in local_modules]
                for imported in modules:
                    top = imported.split('.')[0]
                    if top == 'qf_research':
                        prefixes = ['.'.join(imported.split('.')[:i])
                                    for i in range(1,len(imported.split('.'))+1)]
                        for prefix in prefixes:
                            if prefix not in local_modules:
                                raise ValueError('Unlisted static project import '+prefix+' in '+name)
                        continue
                    if root is not None and (root/'scripts'/(top+'.py')).exists():
                        sibling='scripts/'+top+'.py'
                        if sibling not in blobs:
                            raise ValueError('Unlisted static sibling import '+top+' in '+name)
                    if top in local_top or top in local_base or top in sys.stdlib_module_names or top == '__future__':
                        continue
                    imports.setdefault(imported, set()).add(name)
    return {name: sorted(paths) for name, paths in sorted(imports.items())}



# These test modules load the named files by path rather than ordinary import.
# Each relationship is reviewed explicitly; arbitrary dynamic loading is not
# inferred or executed by the source inventory.
DYNAMIC_SOURCE_DEPENDENCIES = {
    'tests/test_reproduce_paper.py': (
        'scripts/reproduce_paper.py', 'scripts/execute_release_notebooks.py',
    ),
    'tests/test_reproduce_current_study.py': ('scripts/reproduce_current_study.py',),
    'tests/test_current_inventory.py': ('scripts/build_current_inventory.py',),
}


def declared_reader_paths(data):
    """Resolve literal reader constants without importing the reader module.

    Only strings, tuples/lists, prior names and starred sequences are accepted.
    CORE_SOURCES and the optional NATIVE_SOURCES are both required. More
    complicated declarations require an explicit review.
    """
    constants = {}

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    value = literal(element.value)
                    if not isinstance(value, tuple):
                        raise ValueError('Reader source expansion must be a literal sequence')
                    values.extend(value)
                else:
                    values.append(literal(element))
            if not all(isinstance(value, str) for value in values):
                raise ValueError('Reader sources must be literal path strings')
            return tuple(values)
        raise ValueError('Reader source constants need explicit review')

    tree = ast.parse(data.decode('utf-8'), filename='scripts/reproduce_current_study.py')
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not set(names).intersection({'NOTEBOOK', 'NOTEBOOK_SOURCES', 'NATIVE_SOURCES', 'CURRENT_TESTS', 'CORE_SOURCES'}):
            continue
        value = literal(node.value)
        for name in names:
            constants[name] = value
    paths = constants.get('CORE_SOURCES')
    if not isinstance(paths, tuple) or not paths:
        raise ValueError('The current reader must declare a nonempty CORE_SOURCES sequence')
    native = constants.get('NATIVE_SOURCES', ())
    if not isinstance(native, tuple):
        raise ValueError('Native reader sources must be a literal path sequence')
    return tuple(dict.fromkeys((*paths, *native)))


def declared_ci_paths(data):
    """Inspect literal Python entry points in the reviewed workflow.

    This guard covers its present single-line Python commands, not arbitrary
    YAML, shell expansion, generated commands or dependency installation.
    Unsupported Python module invocations require explicit review.
    """
    paths = set()
    for line in data.decode('utf-8').splitlines():
        command = line.strip()
        if command.startswith('run: '):
            command = command[5:]
        if not command.startswith(('python ', 'python3 ')):
            continue
        arguments = shlex.split(command, comments=True)[1:]
        while arguments and arguments[0] in {'-B', '-S', '-u'}:
            arguments.pop(0)
        if not arguments:
            raise ValueError('The workflow Python command needs explicit review')
        if arguments[0].endswith('.py'):
            paths.add(arguments[0])
        elif arguments[:2] == ['-m', 'pip']:
            continue
        elif arguments[:3] == ['-m', 'unittest', 'discover']:
            try:
                folder = arguments[arguments.index('-s')+1]
                pattern = arguments[arguments.index('-p')+1]
            except (ValueError, IndexError) as error:
                raise ValueError('Workflow unittest discovery needs literal -s and -p arguments') from error
            if any(character in folder+pattern for character in '*?[]${}') or not pattern.endswith('.py'):
                raise ValueError('Workflow unittest selection needs an explicit source review')
            paths.add((Path(folder)/pattern).as_posix())
        elif arguments[:2] == ['-m', 'pytest']:
            paths.update(argument for argument in arguments[2:] if argument.endswith('.py'))
            if not any(argument.endswith('.py') for argument in arguments[2:]):
                raise ValueError('Workflow pytest selection needs explicit source paths')
        else:
            raise ValueError('Unsupported workflow Python entry point needs explicit review')
    return tuple(sorted(paths))


def check_declared_entrypoints(blobs):
    """Require the reader, workflow and reviewed dynamic loads to be inventoried."""
    required = {}
    reader = 'scripts/reproduce_current_study.py'
    workflow = '.github/workflows/ci.yml'
    if reader in blobs:
        required[reader] = declared_reader_paths(blobs[reader])
    if workflow in blobs:
        required[workflow] = declared_ci_paths(blobs[workflow])
    for source, dependencies in DYNAMIC_SOURCE_DEPENDENCIES.items():
        if source in blobs:
            required[source] = (*required.get(source, ()), *dependencies)
    for source, dependencies in required.items():
        for dependency in dependencies:
            if dependency not in blobs:
                raise ValueError('Unlisted declared source '+dependency+' required by '+source)


def build_manifest(root=ROOT, *, groups=REVIEWED_GROUPS):
    root = Path(root).resolve(strict=True)
    verifier = load_verifier(root)
    selection = reviewed_selection(groups, verifier)
    blobs = {name: verifier.checked_file(root,name).read_bytes() for name in selection}
    check_declared_entrypoints(blobs)
    document = dict(
        schema_version=1,
        title='Current public-model research source inventory',
        scope='Reviewed source bytes for the current manuscript, public-model aggregate checks, notebooks and numerical research implementation; no provider observations or private study ledgers.',
        dependency_scope='Explicit reviewed file selection with static Python import checks and syntax-only guards for declared reader sources, literal CI Python entry points and named dynamic test loads. Other dynamic imports, compiled dependencies, installed packages, mathematical correctness and complete dependency coverage are not established by this inventory.',
        files=[dict(path=name, category=category, sha256=hashlib.sha256(blobs[name]).hexdigest(),
                    bytes=len(blobs[name])) for name,category in selection.items()],
        external_imports=external_imports(blobs, root=root),
    )
    verifier.validate_manifest(document, MANIFEST)
    for name,data in blobs.items():
        if verifier.checked_file(root,name).read_bytes() != data:
            raise ValueError('Reviewed source changed during inventory construction: '+name)
    return document


def encoded_manifest(document):
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def write_manifest(root, document):
    """Write only the new inventory, rejecting symlinked output components."""
    root = Path(root).resolve(strict=True)
    verifier = load_verifier(root)
    verifier.validate_manifest(document, MANIFEST)
    parent = root/'replication'
    destination = root/MANIFEST
    if parent.is_symlink() or not parent.is_dir() or destination.is_symlink():
        raise ValueError('The current inventory destination must be a regular local path')
    if destination.exists() and not destination.is_file():
        raise ValueError('The current inventory destination is not a regular file')
    for entry in document['files']:
        data = verifier.checked_file(root,entry['path']).read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('Reviewed source changed before inventory write: '+entry['path'])
    with destination.open('wb') as stream:
        stream.write(encoded_manifest(document))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write', action='store_true', help='Write only the new current inventory after review')
    args = parser.parse_args(argv)
    try:
        document = build_manifest(args.root)
        if args.write:
            write_manifest(args.root, document)
        else:
            sys.stdout.buffer.write(encoded_manifest(document))
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
