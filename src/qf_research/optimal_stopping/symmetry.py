"""Symmetric orbit polynomials and analytical GBM conditional moments.

The complete ordinary, unsorted monomial order is increasing total degree,
with combinations_with_replacement within each degree. Averaging coefficients
over an exponent-pattern orbit is exactly the coordinate-permutation
projection. It is not a conversion of a polynomial evaluated on sorted or
strike-centred coordinates. Coordinates here are already non-negative and
may be positively normalised by the caller.

The seven orbit sums are 1, sum x_i, sum x_i**2, sum_{i<j} x_i*x_j,
sum x_i**3, sum_{i!=j} x_i**2*x_j, and sum_{i<j<k} x_i*x_j*x_k.
Positive prefix recurrences avoid subtracting nearly equal power sums.

Projection and orbit_basis_exact use exact rational arithmetic on the
supplied integers, rational values and finite binary floats. Conditional
moments use analytical lognormal identities evaluated numerically; they are
not exact numerical moment or rounding-error certificates.
"""
from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations_with_replacement
import math
from numbers import Integral, Real


ORBIT_NAMES=('1','m1','m2','m11','m3','m21','m111')
EXACT_SCOPE='Exact rational projection and orbit sums of the supplied finite numeric values; no transcendental moment evaluation.'
FLOATING_SCOPE='Analytical GBM identities evaluated with binary floating arithmetic, math.exp and optional NumPy; no outward enclosure or Gaussian-law certificate. Non-finite or underflowing positive intermediates raise ArithmeticError.'


def _integer(value,name,minimum):
    if type(value) is not int or value<minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return value


def _exact(value,name):
    if isinstance(value,bool):
        raise ValueError(f'{name} cannot be boolean')
    if isinstance(value,Fraction):return value
    if isinstance(value,Integral):return Fraction(int(value))
    if isinstance(value,Real) and hasattr(value,'as_integer_ratio'):
        try:return Fraction(*value.as_integer_ratio())
        except (OverflowError,ValueError,TypeError) as error:
            raise ValueError(f'{name} must be finite') from error
    raise ValueError(f'{name} must be an integer, rational or finite binary float')


def _values(value,name):
    if isinstance(value,(str,bytes,bytearray,Mapping,Set)):
        raise ValueError(f'{name} must be an ordered numeric sequence')
    try:return tuple(value)
    except TypeError as error:raise ValueError(f'{name} must be an ordered numeric sequence') from error


def _vector(values,name,*,nonnegative=False):
    answer=tuple(_exact(value,f'{name}[{index}]') for index,value in enumerate(_values(values,name)))
    if not answer or (nonnegative and any(value<0 for value in answer)):
        raise ValueError(f'{name} must be non-empty'+(' and non-negative' if nonnegative else ''))
    return answer


def _parameter(values,dimension,name,*,nonnegative=False):
    if isinstance(values,(Integral,Real,Fraction)):
        scalar=_exact(values,name);answer=(scalar,)*dimension
    else:answer=_vector(values,name)
    if len(answer)!=dimension or (nonnegative and any(value<0 for value in answer)):
        raise ValueError(f'{name} must match the state dimension'+(' and be non-negative' if nonnegative else ''))
    return answer


def complete_monomial_indices(dimension,degree=3):
    """Return every unsorted coordinate monomial through degree zero to three."""
    dimension=_integer(dimension,'Dimension',1);degree=_integer(degree,'Degree',0)
    if degree>3:raise ValueError('Only degrees zero to three are supported')
    return ((),*(indices for order in range(1,degree+1)
                  for indices in combinations_with_replacement(range(dimension),order)))


def monomial_orbit_index(signature):
    """Map a nondecreasing coordinate-index monomial to its orbit-sum column."""
    signature=_values(signature,'Monomial signature')
    if len(signature)>3 or any(type(index) is not int or index<0 for index in signature) or tuple(sorted(signature))!=signature:
        raise ValueError('A monomial requires zero to three ordered non-negative integer indices')
    order=len(signature)
    if order==0:return 0
    if order==1:return 1
    if order==2:return 2 if signature[0]==signature[1] else 3
    distinct=len(set(signature))
    return {1:4,2:5,3:6}[distinct]


def project_coefficients(coefficients,dimension,degree=3):
    """Exactly average complete unsorted polynomial coefficients within orbits.

    The returned seven Fractions multiply orbit sums, without multinomial
    normalisation. At dimension one m11, m21 and m111 are absent; at dimension
    two m111 is absent. Missing degree groups also receive exactly zero.
    """
    indices=complete_monomial_indices(dimension,degree)
    coefficients=_values(coefficients,'Polynomial coefficients')
    if len(coefficients)!=len(indices):
        raise ValueError('The coefficient count must equal the complete declared monomial basis')
    totals=[Fraction(0) for _ in ORBIT_NAMES];counts=[0 for _ in ORBIT_NAMES]
    for position,(signature,coefficient) in enumerate(zip(indices,coefficients)):
        orbit=monomial_orbit_index(signature)
        totals[orbit]+=_exact(coefficient,f'Coefficient[{position}]');counts[orbit]+=1
    return tuple(total/count if count else Fraction(0) for total,count in zip(totals,counts))


def orbit_basis_exact(state):
    """Compute seven exact orbit sums in a positive single-pass recurrence.

    There are O(d) rational arithmetic operations and a fixed number of
    accumulators. Zero coordinates are valid absorbing GBM states.
    """
    if isinstance(state,(str,bytes,bytearray,Mapping,Set)):
        raise ValueError('State coordinates must be ordered')
    try:iterator=iter(state)
    except TypeError as error:raise ValueError('State coordinates must be an ordered sequence') from error
    first=second=pairs=third=mixed=triples=Fraction(0);count=0
    for count,value in enumerate(iterator,1):
        x=_exact(value,f'State[{count-1}]')
        if x<0:raise ValueError('State coordinates must be non-negative')
        square=x*x
        triples+=x*pairs
        mixed+=square*first+x*second
        pairs+=x*first
        first+=x;second+=square;third+=square*x
    if not count:raise ValueError('A non-empty state vector is required')
    return Fraction(1),first,second,pairs,third,mixed,triples


def orbit_basis_float(states):
    """Vectorise positive prefix sums over the final NumPy state dimension.

    Input may be one state or batched states. No subtraction-based power-sum
    identity or cross-path reduction is used. This is a floating evaluation,
    not an enclosure; overflow and positive-square/cube underflow are errors.
    """
    try:import numpy as np
    except ImportError as error:raise RuntimeError('The optional floating basis requires NumPy') from error
    if not isinstance(states,np.ndarray):
        def inspect(value):
            if isinstance(value,(list,tuple)):
                for item in value:inspect(item)
            else:_exact(value,'Floating state')
        inspect(states)
    raw=np.asarray(states)
    if raw.dtype.kind not in ('f','i','u','O'):
        raise ValueError('Floating states must contain non-boolean numeric coordinates')
    if raw.dtype.kind=='O':
        for value in raw.flat:_exact(value,'Floating state')
    try:states=np.asarray(states,dtype=float)
    except (ValueError,TypeError,OverflowError) as error:raise ValueError('Floating states must be representable finite numbers') from error
    if states.ndim<1 or states.shape[-1]<1 or any(size<1 for size in states.shape) or not np.all(np.isfinite(states)) or np.any(states<0):
        raise ValueError('Require complete finite non-negative states in the last dimension')
    shape=states.shape[:-1]
    first=np.zeros(shape);second=np.zeros(shape);pairs=np.zeros(shape)
    third=np.zeros(shape);mixed=np.zeros(shape);triples=np.zeros(shape)
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        for index in range(states.shape[-1]):
            x=states[...,index];square=x*x;cube=square*x
            if np.any((x>0)&((square==0)|(cube==0))):
                raise ArithmeticError('Positive coordinate powers underflow in the floating basis')
            triples=triples+x*pairs
            mixed=mixed+square*first+x*second
            pairs=pairs+x*first
            first=first+x;second=second+square;third=third+cube
        result=np.stack((np.ones(shape),first,second,pairs,third,mixed,triples),axis=-1)
    if not np.all(np.isfinite(result)):raise ArithmeticError('An orbit sum is not finite in floating arithmetic')
    return result


def _exchangeable_correlation(correlation,dimension):
    if isinstance(correlation,(Integral,Real,Fraction)):
        rho=_exact(correlation,'Correlation')
    else:
        rows=_values(correlation,'Correlation matrix')
        matrix=tuple(_vector(row,'Correlation row') for row in rows)
        if len(matrix)!=dimension or any(len(row)!=dimension for row in matrix):
            raise ValueError('The correlation matrix must match the state dimension')
        if any(matrix[i][i]!=1 for i in range(dimension)) or any(matrix[i][j]!=matrix[j][i] for i in range(dimension) for j in range(dimension)):
            raise ValueError('A symmetric correlation matrix with exact unit diagonal is required')
        rho=matrix[0][1] if dimension>1 else Fraction(0)
        if any(matrix[i][j]!=rho for i in range(dimension) for j in range(dimension) if i!=j):
            raise ValueError('Non-exchangeable correlations are not supported by this moment family')
    lower=-Fraction(1,dimension-1) if dimension>1 else Fraction(-1)
    if not lower<=rho<=1:
        raise ValueError('Exchangeable correlation is outside its positive-semidefinite range')
    return rho


@dataclass(frozen=True)
class MomentLaw:
    dimension: int
    step: Fraction
    rate: Fraction
    dividends: tuple[Fraction,...]
    volatilities: tuple[Fraction,...]
    correlation: Fraction
    family: str

    @property
    def covariance(self):
        """Exact instantaneous covariance, materialised only when requested."""
        return tuple(tuple(self.volatilities[i]*self.volatilities[j]*(1 if i==j else self.correlation)
            for j in range(self.dimension)) for i in range(self.dimension))


def validate_moment_law(dimension,step,*,rate,dividends,volatilities,correlation=0):
    """Validate either supported analytical family using exact parameter values.

    Signed finite rates/dividend yields are allowed. Volatilities and step
    are non-negative. PSD boundary comparisons are exact: use Fraction for
    a singular lower endpoint that is not representable as a binary float.
    Scalar correlation validation needs O(d) parameter work; an explicitly
    supplied dense matrix necessarily requires O(d**2) inspection.
    """
    dimension=_integer(dimension,'Dimension',1);step=_exact(step,'Step');rate=_exact(rate,'Rate')
    if step<0:raise ValueError('The time step must be non-negative')
    dividends=_parameter(dividends,dimension,'Dividends')
    volatilities=_parameter(volatilities,dimension,'Volatilities',nonnegative=True)
    rho=_exchangeable_correlation(correlation,dimension)
    common=all(value==volatilities[0] for value in volatilities)
    if not common and rho!=0:
        raise ValueError('Heterogeneous volatilities require independent shocks in this analytical family')
    family='common_volatility_exchangeable' if common else 'independent_heterogeneous_volatility'
    return MomentLaw(dimension,step,rate,dividends,volatilities,rho,family)


def _float(value,name):
    try:answer=float(value)
    except (OverflowError,ValueError) as error:raise ValueError(f'{name} is not representable as a finite float') from error
    if not math.isfinite(answer) or (value!=0 and answer==0):
        raise ValueError(f'{name} is not representable as a non-underflowing finite float')
    return answer


def _exp(exponent):
    if not math.isfinite(exponent):raise ArithmeticError('An analytical exponent is non-finite')
    try:value=math.exp(exponent)
    except OverflowError as error:raise ArithmeticError('An analytical growth factor overflows') from error
    if value==0 or not math.isfinite(value):raise ArithmeticError('An analytical positive growth factor underflows or is non-finite')
    return value


def _finite_orbits(result):
    if any(not math.isfinite(value) for value in result):
        raise ArithmeticError('An analytical orbit moment is non-finite')
    return tuple(result)


def _positive_product(*values):
    """Reject an unrepresentable non-zero product instead of returning zero."""
    if any(value==0 for value in values):return 0.
    result=1.
    for value in values:
        result*=value
        if result==0 or not math.isfinite(result):
            raise ArithmeticError('A positive analytical product underflows or is non-finite')
    return result


def _float_orbits_one(state):
    first=second=pairs=third=mixed=triples=0.
    for x in state:
        square=x*x;cube=square*x
        if x>0 and (square==0 or cube==0):raise ArithmeticError('Positive coordinate powers underflow in the floating moments')
        triples+=x*pairs;mixed+=square*first+x*second;pairs+=x*first
        first+=x;second+=square;third+=cube
    return _finite_orbits((1.,first,second,pairs,third,mixed,triples))


def conditional_orbit_moments(state,step,*,rate,dividends,volatilities,correlation=0):
    """Numerically evaluate the seven analytical conditional orbit moments.

    Common sigma/exchangeable rho permits heterogeneous dividend yields.
    With z_i=x_i*exp((r-q_i)*h), the seven growth factors are 1,1,
    exp(sigma**2*h), exp(rho*sigma**2*h), exp(3*sigma**2*h),
    exp((1+2*rho)*sigma**2*h), exp(3*rho*sigma**2*h).

    Independent heterogeneous sigma instead streams E[X_i], E[X_i**2]
    and E[X_i**3], plus their independent mixed products. No simulation or
    factorisation of a singular covariance matrix is needed. This method
    reports floating point values, not certified exact numerical moments.
    """
    exact_state=_vector(state,'State',nonnegative=True)
    law=validate_moment_law(len(exact_state),step,rate=rate,dividends=dividends,volatilities=volatilities,correlation=correlation)
    state=tuple(_float(value,'State') for value in exact_state)
    h=_float(law.step,'Step');r=_float(law.rate,'Rate')
    q=tuple(_float(value,'Dividend') for value in law.dividends)
    sigma=tuple(_float(value,'Volatility') for value in law.volatilities)
    rho=_float(law.correlation,'Correlation')
    if h==0:return _float_orbits_one(state)
    if not any(state):return (1.,0.,0.,0.,0.,0.,0.)
    z=tuple(0. if x==0 else x*_exp((r-dividend)*h) for x,dividend in zip(state,q))
    if any(x>0 and grown==0 for x,grown in zip(state,z)) or any(not math.isfinite(value) for value in z):
        raise ArithmeticError('A positive first conditional moment underflows or is non-finite')
    if law.family=='common_volatility_exchangeable':
        base=_float_orbits_one(z)
        variance=sigma[0]*sigma[0]*h
        multipliers=(0.,0.,1.,rho,3.,1.+2.*rho,3.*rho)
        result=tuple(value if column<2 or value==0 else _positive_product(value,_exp(multiplier*variance))
            for column,(value,multiplier) in enumerate(zip(base,multipliers)))
        return _finite_orbits(result)
    first=second=pairs=third=mixed=triples=0.
    for grown,volatility in zip(z,sigma):
        if grown==0:continue
        square=grown*grown;cube=square*grown
        if square==0 or cube==0:raise ArithmeticError('Positive conditional powers underflow before their growth factor')
        variance=volatility*volatility*h
        moment2=square*_exp(variance);moment3=cube*_exp(3.*variance)
        triples+=grown*pairs;mixed+=moment2*first+grown*second;pairs+=grown*first
        first+=grown;second+=moment2;third+=moment3
    return _finite_orbits((1.,first,second,pairs,third,mixed,triples))


def conditional_polynomial_value(coefficients,state,step,**law_parameters):
    """Floating value of a seven-orbit polynomial conditional expectation.

    Exact projected Fractions may be supplied, but coefficient conversion,
    transcendental moments and the signed fsum are floating computations.
    An outward certification wrapper is a separate component.
    """
    coefficients=_values(coefficients,'Orbit coefficients')
    if len(coefficients)!=len(ORBIT_NAMES):raise ValueError('Exactly seven orbit coefficients are required')
    values=tuple(_float(_exact(value,f'Orbit coefficient[{index}]'),'Orbit coefficient') for index,value in enumerate(coefficients))
    moments=conditional_orbit_moments(state,step,**law_parameters)
    try:answer=math.fsum(coefficient*moment for coefficient,moment in zip(values,moments))
    except (OverflowError,ValueError) as error:raise ArithmeticError('The floating conditional polynomial sum is non-finite') from error
    if not math.isfinite(answer):raise ArithmeticError('The floating conditional polynomial value is non-finite')
    return answer


def _prefix_suffix(first_moments,second_moments):
    """Positive first, second and unordered-pair totals on each side of i."""
    dimension=len(first_moments)
    prefix1=[0.]*(dimension+1);prefix2=[0.]*(dimension+1);prefix11=[0.]*(dimension+1)
    suffix1=[0.]*(dimension+1);suffix2=[0.]*(dimension+1);suffix11=[0.]*(dimension+1)
    for index,(first,second) in enumerate(zip(first_moments,second_moments)):
        prefix11[index+1]=prefix11[index]+first*prefix1[index]
        prefix1[index+1]=prefix1[index]+first;prefix2[index+1]=prefix2[index]+second
    for index in range(dimension-1,-1,-1):
        first,second=first_moments[index],second_moments[index]
        suffix11[index]=suffix11[index+1]+first*suffix1[index+1]
        suffix1[index]=suffix1[index+1]+first;suffix2[index]=suffix2[index+1]+second
    return prefix1,prefix2,prefix11,suffix1,suffix2,suffix11


def conditional_polynomial_gradient(coefficients,state,step,*,rate,dividends,volatilities,correlation=0):
    """Floating analytical gradient with respect to already normalised x.

    Positive prefix/suffix sums compute the other-coordinate totals without
    subtracting a large current term from a nearly equal power sum. For a
    caller using x=S/scale, the derivative with respect to S is this result
    divided by scale. If the polynomial is a time-zero discounted conditional
    value, its ideal-model asset holding at time t is exp(r*t)/scale times
    this normalised-coordinate gradient.
    No gradient rounding-error certificate is asserted.
    """
    coefficients=_values(coefficients,'Orbit coefficients')
    if len(coefficients)!=7:raise ValueError('Exactly seven orbit coefficients are required')
    coefficients=tuple(_float(_exact(value,f'Orbit coefficient[{index}]'),'Orbit coefficient') for index,value in enumerate(coefficients))
    exact_state=_vector(state,'State',nonnegative=True)
    law=validate_moment_law(len(exact_state),step,rate=rate,dividends=dividends,volatilities=volatilities,correlation=correlation)
    state=tuple(_float(value,'State') for value in exact_state)
    h=_float(law.step,'Step');r=_float(law.rate,'Rate')
    q=tuple(_float(value,'Dividend') for value in law.dividends)
    sigma=tuple(_float(value,'Volatility') for value in law.volatilities)
    rho=_float(law.correlation,'Correlation')
    growth=(1.,)*len(state) if h==0 else tuple(_exp((r-dividend)*h) for dividend in q)
    first=tuple(x*factor for x,factor in zip(state,growth))
    if any(x>0 and moment==0 for x,moment in zip(state,first)):
        raise ArithmeticError('A positive first moment underflows in the gradient')
    if law.family=='common_volatility_exchangeable':
        _float_orbits_one(first)
        second=tuple(value*value for value in first)
        variance=0. if h==0 else sigma[0]*sigma[0]*h
        factor_exponents=(0.,0.,1.,rho,3.,1.+2.*rho,3.*rho)
        second_growth=third_growth=None
    else:
        variance=tuple(0. if h==0 else value*value*h for value in sigma)
        second_growth=(1.,)*len(state) if h==0 else tuple(_exp(2.*(r-dividend)*h+v) for dividend,v in zip(q,variance))
        third_growth=(1.,)*len(state) if h==0 else tuple(_exp(3.*(r-dividend)*h+3.*v) for dividend,v in zip(q,variance))
        second=tuple(x*x*factor for x,factor in zip(state,second_growth))
        if any(x>0 and moment==0 for x,moment in zip(state,second)):
            raise ArithmeticError('A positive second moment underflows in the gradient')
    if any(not math.isfinite(value) for value in (*first,*second)):
        raise ArithmeticError('A gradient moment is non-finite')
    p1,p2,p11,s1,s2,s11=_prefix_suffix(first,second)
    result=[]
    for index,x in enumerate(state):
        other1=p1[index]+s1[index+1];other2=p2[index]+s2[index+1]
        other11=p11[index]+s11[index+1]+p1[index]*s1[index+1]
        if law.family=='common_volatility_exchangeable':
            z=first[index]
            features=(0.,1.,2.*z,other1,3.*z*z,2.*z*other1+other2,other11)
            derivatives=tuple(0. if value==0 else _positive_product(value,growth[index],1. if column<2 else _exp(factor_exponents[column]*variance))
                for column,value in enumerate(features))
        else:
            g1,g2,g3=growth[index],second_growth[index],third_growth[index]
            derivatives=(0.,g1,_positive_product(2.,x,g2),_positive_product(g1,other1),_positive_product(3.,x,x,g3),
                _positive_product(2.,x,g2,other1)+_positive_product(g1,other2),_positive_product(g1,other11))
        if any(not math.isfinite(value) for value in derivatives):
            raise ArithmeticError('An analytical gradient feature is non-finite')
        try:value=math.fsum(coefficient*derivative for coefficient,derivative in zip(coefficients,derivatives))
        except (OverflowError,ValueError) as error:raise ArithmeticError('An analytical polynomial gradient is non-finite') from error
        if not math.isfinite(value):raise ArithmeticError('An analytical polynomial gradient is non-finite')
        result.append(value)
    return tuple(result)


def _rational_record(value):
    try:approximation=float(value)
    except OverflowError:approximation=None
    if approximation is not None and not math.isfinite(approximation):approximation=None
    return dict(numerator=str(value.numerator),denominator=str(value.denominator),approximation=approximation)


def _arb_interval_record(value):
    if not value.is_finite():raise ArithmeticError('A non-finite Arb value cannot certify an analytical quantity')
    lower,upper=value.lower().fmpq(),value.upper().fmpq()
    lower=Fraction(int(lower.p),int(lower.q));upper=Fraction(int(upper.p),int(upper.q))
    if lower>upper:raise ArithmeticError('An analytical certificate has invalid endpoint order')
    return dict(lower=_rational_record(lower),upper=_rational_record(upper),width=_rational_record(upper-lower))


def certify_conditional_polynomial(coefficients,state,step,*,rate,dividends,volatilities,correlation=0,precision=128):
    """Enclose analytical value and normalised-coordinate gradient with Arb.

    Supplied binary floats mean their exact rational values. The certificates
    concern the supported ideal GBM kernel at those exact input states, not
    the distribution of a floating simulator or an unrounded model path.
    For x=S/scale, divide each gradient certificate by that positive scale
    to obtain the corresponding S derivative. A time-zero discounted
    conditional value has ideal-model asset holding exp(r*t)/scale times
    this gradient at time t. No sampling coverage or exercise-policy value
    improvement is asserted.
    """
    precision=_integer(precision,'Precision',32)
    coefficients=_values(coefficients,'Orbit coefficients')
    if len(coefficients)!=7:raise ValueError('Exactly seven orbit coefficients are required')
    coefficients=tuple(_exact(value,f'Orbit coefficient[{index}]') for index,value in enumerate(coefficients))
    state=_vector(state,'State',nonnegative=True)
    law=validate_moment_law(len(state),step,rate=rate,dividends=dividends,volatilities=volatilities,correlation=correlation)
    try:from flint import arb,ctx,fmpq
    except ImportError as error:raise RuntimeError('Analytical enclosures require the separate python-flint environment') from error
    def ball(value):return arb(fmpq(value.numerator,value.denominator))
    def zero():return arb(0)
    with ctx.workprec(precision):
        x=tuple(ball(value) for value in state);h=ball(law.step)
        first_growth=tuple((ball(law.rate-dividend)*h).exp() for dividend in law.dividends)
        first=tuple(value*factor for value,factor in zip(x,first_growth))
        if law.family=='common_volatility_exchangeable':
            second=tuple(value*value for value in first)
            third=tuple(value*value*value for value in first)
            variance=ball(law.volatilities[0]**2*law.step)
            exponents=(Fraction(0),Fraction(0),Fraction(1),law.correlation,Fraction(3),1+2*law.correlation,3*law.correlation)
            factors=tuple((ball(exponent)*variance).exp() for exponent in exponents)
            second_growth=third_growth=None
        else:
            second_growth=tuple(ball(2*(law.rate-dividend)*law.step+volatility**2*law.step).exp()
                for dividend,volatility in zip(law.dividends,law.volatilities))
            third_growth=tuple(ball(3*(law.rate-dividend)*law.step+3*volatility**2*law.step).exp()
                for dividend,volatility in zip(law.dividends,law.volatilities))
            second=tuple(value*value*factor for value,factor in zip(x,second_growth))
            third=tuple(value*value*value*factor for value,factor in zip(x,third_growth))
            factors=tuple(arb(1) for _ in ORBIT_NAMES)
        s1=zero();s2=zero();s11=zero();s3=zero();s21=zero();s111=zero()
        for a,b,c in zip(first,second,third):
            s111=s111+a*s11;s21=s21+b*s1+a*s2;s11=s11+a*s1
            s1=s1+a;s2=s2+b;s3=s3+c
        moments=tuple(value*factor for value,factor in zip((arb(1),s1,s2,s11,s3,s21,s111),factors))
        value=sum((ball(coefficient)*moment for coefficient,moment in zip(coefficients,moments)),zero())
        dimension=len(state)
        p1=[zero() for _ in range(dimension+1)];p2=[zero() for _ in range(dimension+1)];p11=[zero() for _ in range(dimension+1)]
        s1=[zero() for _ in range(dimension+1)];s2=[zero() for _ in range(dimension+1)];s11=[zero() for _ in range(dimension+1)]
        for index,(a,b) in enumerate(zip(first,second)):
            p11[index+1]=p11[index]+a*p1[index];p1[index+1]=p1[index]+a;p2[index+1]=p2[index]+b
        for index in range(dimension-1,-1,-1):
            a,b=first[index],second[index]
            s11[index]=s11[index+1]+a*s1[index+1];s1[index]=s1[index+1]+a;s2[index]=s2[index+1]+b
        gradients=[]
        for index,current in enumerate(x):
            other1=p1[index]+s1[index+1];other2=p2[index]+s2[index+1]
            other11=p11[index]+s11[index+1]+p1[index]*s1[index+1]
            if law.family=='common_volatility_exchangeable':
                z=first[index]
                features=(zero(),arb(1),2*z,other1,3*z*z,2*z*other1+other2,other11)
                derivatives=tuple(feature*first_growth[index]*factor for feature,factor in zip(features,factors))
            else:
                g1,g2,g3=first_growth[index],second_growth[index],third_growth[index]
                derivatives=(zero(),g1,2*current*g2,g1*other1,3*current*current*g3,
                    2*current*g2*other1+g1*other2,g1*other11)
            gradient=sum((ball(coefficient)*derivative for coefficient,derivative in zip(coefficients,derivatives)),zero())
            gradients.append(_arb_interval_record(gradient))
        result=dict(schema_version=1,precision_bits=precision,python_flint_version=__import__('flint').__version__,
            law_family=law.family,state_dimension=dimension,value_certificate=_arb_interval_record(value),gradient_certificates=gradients,
            analytical_value_enclosed=True,analytical_gradient_enclosed=True,supplied_numeric_inputs_treated_as_exact=True,
            gaussian_sampling_law_certified=False,unrounded_ideal_path_enclosure_established=False,statistical_coverage_certified=False,
            gradient_coordinates='Already normalised x; the caller applies the 1/scale chain factor for x=S/scale.',
            scope='Outward analytical polynomial conditional value and gradient under the declared ideal GBM kernel at exact supplied inputs. No simulation, fitting, policy improvement or sampling-law certificate.')
    return result
