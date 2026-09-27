"""Count author reference-graph components, without fetching any FFHQ images."""
import ast
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_integrity import ROOT, sha, write_new


def main():
    base = ROOT / 'outputs/ffhq_ref_metadata_v1'
    receipt = json.loads((base/'receipt.json').read_text())
    for item in receipt['files']:
        assert sha(base/item['name']) == item['sha256']
    splits, summaries = {}, {}
    for split in ['train', 'val', 'test']:
        names = (base/'id_based_ffhq_split'/f'{split}_image.txt').read_text().splitlines()
        assert len(set(names)) == len(names)
        splits[split] = set(names)
        parent = {n: n for n in names}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def merge(a, b):
            parent[find(a)] = find(b)
        targets, violations = set(), []
        with (base/'reference_mapping'/f'{split}_references.csv').open(newline='') as stream:
            mappings = list(csv.DictReader(stream))
        for row in mappings:
            target = row['gt_image']
            assert target not in targets
            targets.add(target)
            refs = ast.literal_eval(row['ref_image'])
            assert isinstance(refs, list) and all(isinstance(r, str) for r in refs)
            for ref in refs:
                if target not in parent or ref not in parent:
                    violations.append({'target': target, 'reference': ref})
                else:
                    merge(target, ref)
        sizes = Counter(find(n) for n in names)
        distribution = Counter(sizes.values())
        summaries[split] = {'listed_images': len(names), 'mapping_rows': len(mappings),
                            'graph_components': len(sizes), 'component_size_distribution': dict(sorted(distribution.items())),
                            'components_at_least_8_photos': sum(n >= 8 for n in sizes.values()),
                            'photos_in_components_at_least_8': sum(n for n in sizes.values() if n >= 8),
                            'cross_split_or_missing_reference_edges': len(violations)}
        assert not violations, 'Unexpected split edges; inspect metadata before any experimental use'
    overlaps = {f'{a}_{b}': len(splits[a]&splits[b]) for a, b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlaps.values())
    result = {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'splits': summaries,
              'image_filename_overlap_between_splits': overlaps, 'metadata_receipt_sha256': sha(base/'receipt.json'),
              'script_sha256': sha(__file__), 'image_pixels_opened': False, 'experiment_people_selected': False,
              'interpretation': 'Connected components of author-provided ArcFace-predicted reference links, not verified identities. Eight photos allow one target, four references and three gallery images before quality/duplicate exclusions. Author train data are not an independent ReF-LDM evaluation. Keep author test data unopened; val is only a possible development source. Cross-dataset identity/pretraining overlap is unknown.'}
    write_new(ROOT/'research/ffhq_ref_pool_v1.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
