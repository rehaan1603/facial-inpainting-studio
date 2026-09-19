def score_reference(record, demand, weights):
    components={
        'regional_quality':sum(demand[k]*record['regions'][k]['quality'] for k in demand),
        'regional_visibility_proxy':sum(demand[k]*record['regions'][k]['visibility_proxy'] for k in demand),
        'regional_exposure':sum(demand[k]*record['regions'][k]['exposure'] for k in demand),
        'global_quality':record['global_quality'],
        'pose_similarity':record['pose_similarity'],
        'identity_compatibility':(record['identity_compatibility']+1)/2,
    }
    return sum(weights[k]*components[k] for k in weights),components
