"""Photo-free export of prerequisite tests, failed probes and diagnostic."""
import json
from pathlib import Path
from spatial_qv_common_v1 import ROOT, pin, read, verify, require


def main():
    paths = {
        'cpu_v1': 'outputs/masked_spatial_qv_cpu_tests_v1.json',
        'tests_v2': 'outputs/masked_spatial_qv_tests_v2.json',
        'probe_v1': 'outputs/masked_spatial_qv_probe_v1/receipt.json',
        'probe_v2': 'outputs/masked_spatial_qv_probe_v2/receipt.json',
        'supervisor_v1': 'outputs/masked_spatial_qv_probe_supervisor_v1/receipt.json',
        'supervisor_v2': 'outputs/masked_spatial_qv_probe_supervisor_v2/receipt.json',
        'diagnostic': 'outputs/spatial_qv_gradient_diagnostic_v1/receipt.json',
        'diagnostic_supervisor': 'outputs/spatial_qv_gradient_diagnostic_supervisor_v1/receipt.json',
    }
    evidence = {}
    for name, relative in paths.items():
        file = ROOT/relative
        record = read(file)
        require(record['status'] != 'running', 'Cannot export unfinished work')
        for source in record.get('source_pins', []):
            verify(source)
        evidence[name] = dict(source=pin(file), record=record)
    output = ROOT/'research/spatial_qv_prerequisite_evidence_v1.json'
    require(not output.exists(), 'Refuse evidence overwrite')
    output.write_bytes((json.dumps(dict(scope='Training prerequisites only; no new reconstruction quality result',
        training_started=False, new_checkpoints=0, evidence=evidence), indent=2, allow_nan=False)+'\n').encode())
    print(output)


if __name__ == '__main__': main()
