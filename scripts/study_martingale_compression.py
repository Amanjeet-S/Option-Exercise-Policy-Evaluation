"""Evaluate compression energy for retained functions under published GBM models.

I hold the previously fitted functions fixed. This is a population-moment
compression study, rather than a new exercise-policy or market-price estimate.
The asymmetric volatility specification follows Becker, Cheridito and Jentzen
(2019), Deep Optimal Stopping, Section 4.1 and Table 2.
"""
from argparse import ArgumentParser
from datetime import datetime, timezone
from fractions import Fraction
import hashlib, importlib.util, json
from pathlib import Path
import sys,time

SOURCE_NAMES=('src/qf_research/optimal_stopping/symmetry.py','src/qf_research/optimal_stopping/martingale_compression.py','tests/test_symmetry.py','tests/test_martingale_compression.py','scripts/study_martingale_compression.py','tests/test_study_martingale_compression.py')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def load(root):
    path=root/'src/qf_research/optimal_stopping/martingale_compression.py'
    spec=importlib.util.spec_from_file_location('compression_study_core',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

REFERENCE_DESIGN_SHA='e4a5bf17934f3f4314dcae94ba8e06e3b259c3f051043214a309f5d530ba3149'
REFERENCE_DESIGN_PATH=None
MODEL_CASES=('exchangeable_reference','published_asymmetric_volatility')
def snapshot(root,retained):
    design_path=REFERENCE_DESIGN_PATH or root/'data/private/revision/2026-10-05/financial_decision_v1/study/design.json'
    if sha(design_path)!=REFERENCE_DESIGN_SHA:raise ValueError('The previously reviewed reference design changed')
    reference=read(design_path)
    if [row['index'] for row in reference['fits']]!=list(range(8)) or [row['root_seed'] for row in reference['fits']]!=list(range(2026100410,2026100418)):raise ValueError('The complete reviewed teacher cohort is required')
    if {name:sha(retained/name) for name in reference['retained_input_sha256']}!=reference['retained_input_sha256']:raise ValueError('A reviewed teacher input changed')
    for i in range(8):
        folder=retained/f'fit_{i:02d}'; seal=read(folder/'assessment_attempt.json')
        if sha(folder/'polynomial_view.json')!=seal['polynomial_view_sha256'] or sha(folder/'inputs.json')!=seal['input_sha256']:raise ValueError('A previously sealed teacher coefficient or input file changed')
    original=reference['source_sha256']
    if {name:sha(root/name) for name in original}!=original:raise ValueError('An earlier registered numerical source changed')
    names=[f'fit_{i:02d}/{name}' for i in range(8) for name in ('inputs.json','polynomial_view.json')]
    return dict(reference_design_sha256=sha(design_path),source_sha256={name:sha(root/name) for name in SOURCE_NAMES},original_registered_source_sha256=original,retained_input_sha256={name:sha(retained/name) for name in names})

def teacher(root,retained,i,core):
    folder=retained/f'fit_{i:02d}';data=read(folder/'inputs.json');view=read(folder/'polynomial_view.json');polynomial=view['degrees']['3']
    model=data['model'];d=len(model['spots']);times=tuple(Fraction(t) for t in model['exercise_times'])
    expected=dict(spots=[100.]*5,volatilities=[.2]*5,dividends=[.1]*5,correlation=[[float(a==b) for b in range(5)] for a in range(5)],rate=.05,strike=100.,exercise_times=[3*j/9 for j in range(10)])
    if model!=expected or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('The declared retained reference model and calendar are required')
    if data['design']['root_seed']!=2026100410+i or view['root_seed']!=2026100410+i:raise ValueError('The declared retained fitting root changed')
    signatures=core.complete_monomial_indices(d,3)
    powers=[[signature.count(k) for k in range(d)] for signature in signatures]
    if polynomial['basis_exponents']!=powers or len(polynomial['coefficients'])!=10 or any(len(x)!=len(signatures) for x in polynomial['coefficients']):raise ValueError('The retained coefficient basis or date counts changed')
    initial=tuple(Fraction(x)/Fraction(model['strike']) for x in model['spots'])
    return model,times,initial,polynomial


def validate_prepared(root,retained,prepared,core):
    cases=prepared['cases']
    if len(cases)!=16 or [(x['fit_index'],x['model_case']) for x in cases]!=[(i,name) for i in range(8) for name in MODEL_CASES]:raise ValueError('Every ordered teacher and model case must be retained exactly once')
    for row in cases:
        model,times,initial,polynomial=teacher(root,retained,row['fit_index'],core)
        vol=tuple(Fraction(x) for x in model['volatilities']) if row['model_case']==MODEL_CASES[0] else tuple(Fraction(2,25)+Fraction(8,25)*i/4 for i in range(5))
        expected_model=dict(rate=Fraction(model['rate']),dividends=tuple(Fraction(x) for x in model['dividends']),volatilities=vol,correlation=Fraction(0))
        actual_model={key:tuple(core.decode_rational(x) for x in value) if isinstance(value,list) else core.decode_rational(value) for key,value in row['model'].items()}
        if actual_model!=expected_model or tuple(core.decode_rational(x) for x in row['initial_scaled_state'])!=initial:raise ValueError('A prepared actual model or scaled initial state changed')
        if len(row['dates'])!=8 or [x['date_index'] for x in row['dates']]!=list(range(1,9)):raise ValueError('Every nonterminal polynomial date must be retained exactly once')
        for date in row['dates']:
            j=date['date_index'];c=tuple(Fraction(x) for x in polynomial['coefficients'][j])
            if tuple(core.decode_rational(x) for x in date['retained_coefficients'])!=c or core.decode_rational(date['previous_time'])!=times[j-1] or core.decode_rational(date['current_time'])!=times[j]:raise ValueError('A prepared coefficient or time row changed its retained teacher')
            if tuple(core.decode_rational(x) for x in date['averaging_coefficients'])!=core.project_coefficients(c,5,3):raise ValueError('Exact averaging no longer matches the retained teacher')
            if len(date['nominal_projection']['frozen_orbit_coefficients'])!=7:raise ValueError('A nominal candidate must retain seven coefficients')
    return True


def prepare(root,retained,output,core):
    before=snapshot(root,retained);write(output/'attempt.json',dict(author='Amanjeet Singh',stage='projection',started_at_utc=datetime.now(timezone.utc).isoformat(),source_snapshot_before=before))
    records=[]
    for i in range(8):
        model,times,initial,polynomial=teacher(root,retained,i,core);d=5
        for name,vol in [('exchangeable_reference',tuple(Fraction(x) for x in model['volatilities'])),('published_asymmetric_volatility',tuple(Fraction(2,25)+Fraction(8,25)*Q/Fraction(d-1) for Q in range(d)))]:
            law=dict(rate=Fraction(model['rate']),dividends=tuple(Fraction(x) for x in model['dividends']),volatilities=vol,correlation=Fraction(0))
            dates=[]
            for j in range(1,len(times)-1):
                c=polynomial['coefficients'][j]
                average=core.project_coefficients(c,d,3)
                fit=core.fit_energy_projection(c,initial,times[j-1],times[j],**law)
                dates.append(dict(date_index=j,previous_time=core.rational_record(times[j-1]),current_time=core.rational_record(times[j]),retained_coefficients=[core.rational_record(Fraction(x)) for x in c],averaging_coefficients=[core.rational_record(x) for x in average],nominal_projection=fit.record()))
            records.append(dict(fit_index=i,model_case=name,initial_scaled_state=[core.rational_record(x) for x in initial],model={key:[core.rational_record(x) for x in value] if isinstance(value,tuple) else core.rational_record(value) for key,value in law.items()},dates=dates))
            print('Prepared fit',i,name,flush=True)
    after=snapshot(root,retained)
    if before!=after:raise ValueError('Source or teacher inputs changed during projection')
    result=dict(author='Amanjeet Singh',completed=True,cases=records,source_snapshot_before=before,source_snapshot_after=after,models=2,teachers=8,nonterminal_dates=8,terminal_payoff_retained=True,new_sampling_executed=False,new_policy_fitting_executed=False,source='Becker, Cheridito and Jentzen (2019), Section 4.1: https://jmlr.org/papers/volume20/18-232/18-232.pdf',scope='The same retained homogeneous-model functions are assessed under the reference law and published asymmetric volatility law. They are not refitted for the latter. This tests compression distortion of the retained functions, not absolute pricing accuracy or observed market performance.')
    validate_prepared(root,retained,result,core)
    write(output/'prepared.json',result)
    return result

def certify(root,retained,preparation,output,core,expected_preparation_sha256):
    if sha(preparation/'prepared.json')!=expected_preparation_sha256:raise ValueError('The sealed preparation bytes changed before certification')
    before=snapshot(root,retained);prepared=read(preparation/'prepared.json')
    validate_prepared(root,retained,prepared,core)
    if not prepared['completed'] or before!=prepared['source_snapshot_after']:raise ValueError('Preparation bindings changed')
    write(output/'attempt.json',dict(author='Amanjeet Singh',stage='outward_energy',started_at_utc=datetime.now(timezone.utc).isoformat(),source_snapshot_before=before,prepared_sha256=sha(preparation/'prepared.json')))
    cases=[]
    for row in prepared['cases']:
        model={key:tuple(core.decode_rational(x) for x in value) if isinstance(value,list) else core.decode_rational(value) for key,value in row['model'].items()}
        initial=tuple(core.decode_rational(x) for x in row['initial_scaled_state'])
        energies={'averaging':[],'nominal_projection':[]}
        for date in row['dates']:
            c=tuple(core.decode_rational(x) for x in date['retained_coefficients'])
            t0,t1=core.decode_rational(date['previous_time']),core.decode_rational(date['current_time'])
            candidates={'averaging':date['averaging_coefficients'],'nominal_projection':date['nominal_projection']['frozen_orbit_coefficients']}
            for name,coefficients in candidates.items():
                b=tuple(core.decode_rational(x) for x in coefficients)
                energies[name].append(core.certify_residual_energy(c,b,initial,t0,t1,**model,precision=128))
        combined={name:core.combine_energy_certificates(value,precision=128) for name,value in energies.items()}
        old=core.decode_interval(combined['averaging']['terminal_energy_certificate']);new=core.decode_interval(combined['nominal_projection']['terminal_energy_certificate'])
        strict=new[1]<old[0]
        if strict and row['model_case']=='exchangeable_reference':raise ArithmeticError('A certified reduction contradicts exact-symmetry optimality of averaging')
        chosen='nominal_projection' if strict and row['model_case']!='exchangeable_reference' else 'averaging'
        cases.append(dict(fit_index=row['fit_index'],model_case=row['model_case'],increment_energy_records=energies,combined=combined,strict_energy_reduction_certified=strict,selected_candidate=chosen,selection_scope='Exact averaging is retained under exchangeability; a numerical asymmetric-law candidate is selected only when its outward upper residual energy is below the averaging lower endpoint. No exact numerical optimality is claimed.',reference_hedge_rms_difference_upper=combined[chosen]['absolute_expected_dual_gap_difference_upper']))
        print('Certified fit',row['fit_index'],row['model_case'],'strict reduction',strict,flush=True)
    after=snapshot(root,retained)
    if before!=after or sha(preparation/'prepared.json')!=read(output/'attempt.json')['prepared_sha256']:raise ValueError('A source, teacher input or preparation changed')
    result=dict(schema_version=1,author='Amanjeet Singh',completed=True,cases=cases,source_snapshot_before=before,source_snapshot_after=after,prepared_sha256=sha(preparation/'prepared.json'),increment_certificates=256,teachers=8,model_cases=2,precision_bits=128,terminal_energy_contribution_exact_zero=True,no_new_pricing_integration_required_for_energy=True,new_sampling_executed=False,new_policy_fitting_executed=False,absolute_pricing_error_established=False,market_validation_established=False,substantive_original_research_contribution_established=False,scope='Certified differences from retained reference martingales under the specified actual GBM models. These population energy certificates concern expected upper prices and RMS differences of modelled hedge gains. They do not estimate a new option price or absolute hedge error.')
    write(output/'result.json',result)
    return result

def main():
    global REFERENCE_DESIGN_PATH,REFERENCE_DESIGN_SHA
    p=ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','certify'));p.add_argument('--root',type=Path,required=True);p.add_argument('--retained-study',type=Path,required=True);p.add_argument('--preparation',type=Path);p.add_argument('--preparation-sha256');p.add_argument('--reference-design',type=Path);p.add_argument('--reference-design-sha256');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.reference_design is None)!=(a.reference_design_sha256 is None):raise ValueError('A replacement reference design requires its explicit sealed hash')
    if a.reference_design is not None:
        REFERENCE_DESIGN_PATH=a.reference_design.resolve();REFERENCE_DESIGN_SHA=a.reference_design_sha256
    root=a.root.resolve();private=(root/'data/private').resolve();retained=a.retained_study.resolve();output=a.output.resolve()
    if not retained.is_relative_to(private) or not output.is_relative_to(private) or output.exists() or output.is_relative_to(retained) or retained.is_relative_to(output):raise ValueError('Use retained private inputs and a separate new private output')
    preparation=None
    if a.stage=='certify':
        if a.preparation is None or not a.preparation_sha256:raise ValueError('Certification requires its private preparation and sealed hash')
        preparation=a.preparation.resolve()
        if not preparation.is_relative_to(private) or output.is_relative_to(preparation) or preparation.is_relative_to(output):raise ValueError('Use separate private preparation and output')
    output.mkdir(parents=True);start=(time.process_time(),time.perf_counter())
    try:
        core=load(root)
        if a.stage=='prepare':prepare(root,retained,output,core)
        else:
            certify(root,retained,preparation,output,core,a.preparation_sha256)
        write(output/'execution.json',dict(author='Amanjeet Singh',completed=True,stage=a.stage,cpu_seconds=time.process_time()-start[0],wall_seconds=time.perf_counter()-start[1],scope='Complete stage through result writing; process startup and this final record are excluded.'))
    except Exception as e:
        write(output/'failure.json',dict(author='Amanjeet Singh',completed=False,stage=a.stage,exception_type=type(e).__name__,message=str(e),cpu_seconds=time.process_time()-start[0],wall_seconds=time.perf_counter()-start[1]));raise
if __name__=='__main__':main()
