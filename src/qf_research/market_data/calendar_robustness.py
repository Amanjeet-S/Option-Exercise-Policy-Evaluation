"""Separate exercise-calendar increments from PDE mesh and anchor effects."""
import math

MESHES = ('base', 'space_only', 'time_only', 'fine', 'wide')


def mesh_changes(prices):
    if set(prices) != set(MESHES) or any(isinstance(v,bool) or not math.isfinite(v) or v<0 for v in prices.values()):
        raise ValueError('Require all five finite non-negative mesh prices')
    return _differences(prices)


def _differences(prices):
    return dict(space_at_coarse_time=prices['space_only']-prices['base'],
        space_at_fine_time=prices['fine']-prices['time_only'],
        time_at_coarse_space=prices['time_only']-prices['base'],
        time_at_fine_space=prices['fine']-prices['space_only'],
        interaction=prices['fine']-prices['space_only']-prices['time_only']+prices['base'],
        domain=prices['wide']-prices['fine'])


def compare_calendars(prices, order, retained_close, bid, ask, allowance):
    """Conditional quote margins; no error estimate or limiting value inferred."""
    if len(order)<2 or len(set(order))!=len(order) or set(order)!=set(prices) or 'core_close' not in order:
        raise ValueError('Unique complete calendar order including core_close required')
    if any(isinstance(v,bool) or not math.isfinite(v) for v in (retained_close,bid,ask,allowance)) or not 0<=bid<ask or retained_close<0 or allowance<=0:
        raise ValueError('Finite positive spread, allowance and non-negative prices required')
    changes={name:mesh_changes(prices[name]) for name in order}
    fine_close=prices['core_close']['fine']
    rows=[]
    for name in order:
        p=prices[name]['fine'];c=changes[name]
        rows.append(dict(calendar=name,price=p,mesh_changes=c,
            space_screen=all(abs(c[k])<=allowance for k in ('space_at_coarse_time','space_at_fine_time')),
            time_screen=all(abs(c[k])<=allowance for k in ('time_at_coarse_space','time_at_fine_space')),
            domain_screen=abs(c['domain'])<=allowance,
            quote_margin=min(p-bid,ask-p),quote_with_margin=bid<=p-allowance and p+allowance<=ask,
            exercise_change_from_close=p-fine_close,
            total_change_from_retained=p-retained_close,
            accounting_residual=(p-retained_close)-((p-fine_close)+(fine_close-retained_close))))
    increments=[]
    for left,right in zip(order,order[1:]):
        values={m:prices[right][m]-prices[left][m] for m in MESHES}
        increments.append(dict(start=left,end=right,values=values,
            changes=_differences(values),
            minimum_increment=min(values.values()),ordering_with_allowance=all(v>=-2*allowance for v in values.values()),
            refinement_screen=all(abs(values[m]-values['fine'])<=allowance for m in MESHES)))
    return dict(calendars=rows,increments=increments,anchor_change=fine_close-retained_close,
                anchor_screen=abs(fine_close-retained_close)<=allowance)
