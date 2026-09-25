"""Seed-complete, identity-unit statistics for the frozen context-support screen."""
import itertools
import numpy as np


def identity_means(rows, metric, seeds):
    grouped = {}
    for row in rows:
        values = grouped.setdefault(row['identity'], {})
        if row['seed'] in values:
            raise ValueError('Duplicate identity/seed row')
        score = row.get('evaluation', {})
        value = score.get('metrics', {}).get(metric) if score.get('status') == 'complete' else None
        values[row['seed']] = value
    valid, excluded = {}, []
    for identity, values in grouped.items():
        if set(values) != set(seeds) or any(v is None or not np.isfinite(v) for v in values.values()):
            excluded.append(identity)
        else:
            valid[identity] = float(np.mean(list(values.values())))
    return valid, sorted(excluded)


def contrast(left, right, metric, seeds):
    a, ae = identity_means(left, metric, seeds)
    b, be = identity_means(right, metric, seeds)
    ids = sorted(a.keys() & b.keys())
    delta = {i: a[i] - b[i] for i in ids}
    result = {'identities': len(ids), 'identity_deltas': delta,
              'excluded_identities': sorted(set(ae + be) | (a.keys() ^ b.keys())),
              'mean_delta': None, 'ci95': None, 'p_exact': None}
    if not ids:
        return result
    x = np.array(list(delta.values()))
    rng = np.random.default_rng(20260925)
    exact = np.array([abs(np.mean(x * s)) for s in itertools.product([-1, 1], repeat=len(x))])
    result.update(mean_delta=float(x.mean()),
                  ci95=np.quantile(rng.choice(x, (10000, len(x)), replace=True).mean(1), [.025, .975]).tolist(),
                  p_exact=float(np.mean(exact >= abs(x.mean()) - 1e-12)))
    return result


def holm(contrasts):
    previous = 0.
    for index, key in enumerate(sorted(contrasts, key=lambda k: contrasts[k]['p_exact'] if contrasts[k]['p_exact'] is not None else 1)):
        p = contrasts[key]['p_exact']
        previous = max(previous, min(1., (len(contrasts) - index) * (p if p is not None else 1.)))
        contrasts[key]['p_holm'] = previous

