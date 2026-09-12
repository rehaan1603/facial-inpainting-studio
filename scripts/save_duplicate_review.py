"""Record explicit visual triage and create separate, provisional reviewed manifests."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/near_duplicate_audit'


def main():
    assert hashlib.sha256((OUT/'candidate_pairs.json').read_bytes()).hexdigest()=='d65d50968c4a47cb50bed1bf515865a50a17331271300fe1cbfe5cbca695bc02', 'Candidate index changed; visual decisions require re-review'
    pairs=json.loads((OUT/'candidate_pairs.json').read_text());ranked=json.loads((OUT/'ranked_candidates.json').read_text());records=json.loads((OUT/'indexed_images.json').read_text())
    # Reviewed contact sheets: ranked_review_1..4 and active_review_1..2.
    near={0,1,9,2,4,42,8,10,6,348,11}
    reviewed={p['candidate_id'] for p in ranked[:32]}|{316,188,190,303,158,76,313,374,189,312,492}
    assert near<=reviewed
    decisions=[];exclude=set()
    for index in sorted(reviewed):
        p=pairs[index]
        decision='near_duplicate_photograph' if index in near else 'different_photographs'
        decisions.append({'candidate_id':index,'decision':decision,'a_path':records[p['a']]['image_path'],'b_path':records[p['b']]['image_path'],'basis':'Assistant visual inspection of paired contact-sheet images; photographic similarity judgment, not identity recognition.'})
        if index in near:exclude.update([records[p['a']]['image_path'],records[p['b']]['image_path']])
    manifests={}
    for dataset in ['celebahq','lapa']:
        src=ROOT/'data/manifests'/f'{dataset}_clean.csv';rows=list(csv.DictReader(src.open()));retained=[r for r in rows if r['image_path'] not in exclude]
        dest=ROOT/'data/manifests'/f'{dataset}_reviewed_v1.csv'
        with dest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(retained)
        manifests[dataset]={'before':len(rows),'after':len(retained),'excluded':len(rows)-len(retained),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'path':str(dest.relative_to(ROOT))}
    excluded_hq={r.get('hq_id') for r in records if r['image_path'] in exclude and r['dataset']=='celebahq'}
    impact={}
    for name in ['pilot_clean_v1','benchmark_v2','area_matched_v3_margin12']:
        casepath=ROOT/'outputs'/name/'cases.json';cases=json.loads(casepath.read_text())
        impact[name]=sorted({str(c['hq_id']) for c in cases}&excluded_hq)
    review={'candidate_pairs_sha256':hashlib.sha256((OUT/'candidate_pairs.json').read_bytes()).hexdigest(),'candidates_total':len(pairs),'pairs_reviewed':len(reviewed),'near_duplicate_pairs':len(near),'different_photograph_pairs':len(reviewed-near),'pending_review':len(pairs)-len(reviewed),'decisions':decisions,'excluded_paths':sorted(exclude),'manifests':manifests,'experiment_case_exclusion_overlap':impact,'limits':['Visual triage is provisional; independent review is recommended before publication.','Unreviewed candidates are retained, not assumed safe or duplicate.','Hash screening cannot establish exhaustive absence of transformed duplicates.','Original manifests and source images are unchanged.','Future runs must explicitly select reviewed manifests; historical scripts still reference their original clean manifests.']}
    (ROOT/'research/near_duplicate_review.json').write_text(json.dumps(review,indent=2))
    print(json.dumps({k:v for k,v in review.items() if k not in ['decisions','excluded_paths','limits']},indent=2))


if __name__=='__main__':main()
