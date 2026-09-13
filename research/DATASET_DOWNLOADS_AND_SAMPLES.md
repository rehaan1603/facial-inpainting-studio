# Dataset downloads and local samples

The repository contains code, recorded selections, hashes and measurements. Face photos and derived image outputs remain local. On 13 September 2026 the project owner selected official downloads and local restoration because there is no separate dataset-owner permission for a GitHub photo mirror.

## Official downloads

| Dataset | Official download entry point | Files used locally |
|---|---|---|
| CelebAMask-HQ | [Author repository and dataset download](https://github.com/switchablenorms/CelebAMask-HQ) | CelebA-HQ-img, parsing masks and CelebA-HQ-to-CelebA-mapping.txt |
| CelebA | [Author download page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) | Anno identity/attribute/landmark records and Eval/list_eval_partition.txt; original CelebA images are not required to restore the HQ reference examples |
| LaPa | [Author repository and download links](https://github.com/jd-opensource/lapa-dataset) | Extracted train, val and test folders with images and annotations |

Follow each author's dataset agreement before downloading. CelebA and CelebAMask-HQ restrict further copying and redistribution, with an internal single-site exception. LaPa's README describes noncommercial dataset use; the repository's software license should not be assumed to resolve every photo's redistribution rights. Links are source entry points rather than temporary Google Drive download tokens.

Copy configs/local.example.json to configs/local.json and set the extracted folders. The HQ root must contain CelebA-HQ-img; the LaPa root must contain train, val and test. Original datasets are not modified by restoration.

## Exact reference inputs

The current website demonstration has one target and four references: HQ IDs 6747, 8910, 12181, 14926 and 16323, respectively, under supplied identity label 916. See [its frozen manifest](../outputs/reference_examples/identity_916/manifest.json).

The twelve-person development diagnostic contains 60 additional photos, with target/reference roles and source/processed hashes in [its frozen manifest](../outputs/reference_diagnostics_v1/manifest.json). These are development examples, not a new untouched test set.

After downloading HQ and installing the recorded base environment, run from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 scripts/restore_reference_samples.py --check-only
.venv/Scripts/python.exe -X utf8 scripts/restore_reference_samples.py
```

Alternatively pass `--hq "C:/your/path/CelebAMask-HQ"`. Restoration uses the already selected photos and recorded mask polygons; no detector, GPU, or reselection is required. It validates all 65 source hashes and all 91 reconstructed PNG hashes before writing. Missing files are created; matching existing files are kept; differing files cause an error. A different Pillow/PNG encoder can fail byte verification: use requirements-lock.txt rather than bypassing the hash check.

This restores targets, four references per case, synthetic masks and occluded inputs. It does not manufacture saved model outputs. Generated outputs require the recorded models, environment and inference settings. Target.png is for evaluation only and must never be passed as a reference to its own reconstruction.

## Earlier samples and all results

[LOCAL_IMAGE_INVENTORY.json](LOCAL_IMAGE_INVENTORY.json) records the paths, sizes and SHA256 hashes of all local output images present when indexed, including historical trials and previews. It does not contain the images themselves. Refresh with `scripts/index_local_images.py` after new runs.

Earlier single-image input selections and source hashes are in [data/manifests](../data/manifests). [REPRODUCE_EARLIER_STAGES.md](REPRODUCE_EARLIER_STAGES.md) records their construction commands. The original failed reference smoke selection is retained in outputs/reference_smoke/manifest.json; it is historical and is not the current website sample.

[REPOSITORY_ARTIFACT_INVENTORY.json](REPOSITORY_ARTIFACT_INVENTORY.json) indexes committed research artifacts. [REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md](REFERENCE_DIAGNOSTIC_PROTOCOL_V3.md) defines the completed 48-candidate/96-row diagnostic; the matching results report includes failures and limits. The photo restoration check records exact file hashes separately. Downloading data or reproducing inputs does not establish improved likeness or publication readiness.
