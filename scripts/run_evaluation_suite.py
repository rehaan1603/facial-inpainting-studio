"""Reproduce/resume the extended development suite from the project Python."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    cache=Path(json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8'))['cache'])
    evaluation=cache/'evaluation_env_v1/Scripts/python.exe'
    reference=cache/'reference_env_v2/Scripts/python.exe'
    steps=[(sys.executable,'prepare_extended_evaluation.py',[]),
           (reference,'run_reference_count_ablation.py',[]),
           (evaluation,'evaluate_extended_diagnostics.py',['--suite','original']),
           (evaluation,'evaluate_extended_diagnostics.py',['--suite','count']),
           (sys.executable,'analyze_extended_ablations.py',[]),
           (sys.executable,'verify_extended_evaluation.py',[]),
           (sys.executable,'plot_extended_ablations.py',[])]
    for python,script,args in steps:
        print(f'Running {script}',flush=True)
        subprocess.run([str(python),'-X','utf8',str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
    print('Complete. See research/EXTENDED_ABLATION_RESULTS_V1.md',flush=True)

if __name__=='__main__':
    main()
