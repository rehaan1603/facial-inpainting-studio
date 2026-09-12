# Repository checkpoint

This private repository preserves the implementation and research record as of 12 September 2026. It includes the local browser application, model integration and evaluation scripts, configuration examples, environment locks, research protocols and measured-result reports.

Source datasets, uploaded faces, reference photos, generated face images, downloaded generative/reference-model weights, Python environments, local path configuration and delivery ZIP archives stay on the original laptop. Detailed experiment CSV/JSON/JSONL records, audit manifests, plots without photographs, and all twenty-four project-trained best/latest mask-refiner checkpoint files are included. The repository therefore does not run inference from a fresh clone until the documented environments, permitted models, datasets and local configuration are restored. The trained refiner files include optimizer state in latest.pt where recorded, allowing the same training to resume after restoring the environment and permitted data.

Start the configured application on the original laptop with Start Studio.cmd and open http://127.0.0.1:8765/. The launcher keeps the server running independently after its window closes. The GPU service binds only to 127.0.0.1; this is not an internet-hosted app or a GitHub Pages deployment.

The reference-photo feature uses existing pretrained methods. Earlier measured experiments, unfinished comparisons and publication requirements are distinguished in research/PROJECT_STATUS.md. Research results are not claims of accepted publication or verified hidden-face recovery. Third-party data, code and weights retain their own terms; see research/DATA_AND_LICENSES.md.

The complete supplemental file list, byte sizes and SHA-256 hashes are in research/REPOSITORY_ARTIFACT_INVENTORY.json. Dataset identity labels and original local source paths are retained in research metadata for traceability. The repository remains private. Failed, partial and prepared-only experiments retain their original status; their presence does not mean their evaluations are complete.
