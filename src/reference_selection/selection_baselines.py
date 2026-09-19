import random

POLICIES=('first','random','identity_only','quality_only','all','mask_aware','zero_scale')

def select(analysis, policy='mask_aware', seed=17):
    if policy not in POLICIES:raise ValueError('Unknown selection policy')
    valid=[r for r in analysis['references'] if r['valid']]
    if not valid:return []
    if policy in ('all','zero_scale'):return [r['path'] for r in valid]
    if policy=='random':return [random.Random(seed).choice(valid)['path']]
    key={'identity_only':'identity_compatibility','quality_only':'global_quality','mask_aware':'mask_aware_score'}.get(policy)
    return [(max(valid,key=lambda r:r[key]) if key else valid[0])['path']]
