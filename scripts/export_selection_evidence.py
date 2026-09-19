"""Export numerical audit records only; dataset and derived face images stay local."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT,sha,write_new

def main():
    base=ROOT/'outputs/generalization_v1'
    comparison=json.loads((base/'evaluation.json').read_text())
    manifest=json.loads((base/'cases/manifest.json').read_text())
    cases=[]
    for c in manifest['cases']:
        analysis=json.loads((base/'analysis'/(c['case_id']+'.json')).read_text())
        for r in analysis['references']:r.pop('path',None)
        cases.append({'case_id':c['case_id'],'identity':c['identity'],'condition':c['condition'],'distortion':c['distortion'],
                      'reference_sha256':[r['sha256'] for r in c['references']],
                      'evaluation_only_target_sha256':c['evaluation_only_target']['sha256'],
                      'evaluation_only_gallery_sha256':[g['sha256'] for g in c['evaluation_only_gallery']],
                      'analysis':analysis})
    generations={}
    for row in comparison['rows']:
        key=row['generation_key']
        if key in generations:continue
        record=json.loads((base/'generations'/key/'record.json').read_text());record.pop('output',None)
        generations[key]=record
    write_new(ROOT/'research/generalization_selection_evidence_v1.json',{
        'no_photographs_included':True,'comparison_signature':json.loads((base/'comparison_signature.json').read_text()),
        'evaluation_signature':json.loads((base/'evaluation_signature.json').read_text()),
        'preservation_verification':json.loads((base/'preservation_verification.json').read_text()),
        'model_verification':json.loads((base/'model_verification.json').read_text()),
        'runtime_check_status':json.loads((base/'runtime_checks/checks.json').read_text())['status'],
        'preparation_sources':{name:sha(ROOT/'scripts'/name) for name in ['prepare_unseen_identity_protocol.py','prepare_restoration_cases.py']},
        'cases':cases,'generations':generations,'rows':comparison['rows'],
        'pretrained_model_provenance':['research/osor_downloads.json','research/reference_adapter_provenance.json','research/reference_faces_provenance.json','research/extended_evaluation_provenance_v1.json','research/identity_evaluator_provenance_v1.json','research/PRETRAINING_EXPOSURE_LEDGER.md'],
        'local_evaluation_sha256':sha(base/'evaluation.json')})

if __name__=='__main__':main()
