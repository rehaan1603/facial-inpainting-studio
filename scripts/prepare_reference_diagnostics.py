"""Prepare a deterministic, development-only four-reference diagnostic manifest.

Run in reference_env. CPU face detection only; no inpainting model is imported.
This does not alter any earlier experiment, source photograph, or signed script.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reference_diagnostics_v1"
COUNT = 12
SEED = "reference-diagnostics-v1-17"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranked(value, purpose):
    return hashlib.sha256(f"{SEED}:{purpose}:{value}".encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    manifest_path = ROOT / "data/manifests/celebahq_reviewed_v2.csv"
    rows = list(csv.DictReader(manifest_path.open(newline="")))
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "val":
            groups[row["identity"]].append(row)

    exclusion_sources = {}
    prior_paths = [
        "outputs/pilot/cases.json", "outputs/pilot_clean_v1/cases.json",
        "outputs/benchmark_v2/cases.json", "outputs/benchmark_v2_resshift/cases.json",
        "outputs/area_matched_v3/cases.json", "outputs/area_matched_v3_margin12/cases.json",
    ]
    for relative in prior_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required prior-case exclusion source is missing: {relative}")
        exclusion_sources[relative] = {"sha256": sha(path), "identity_labels": sorted({str(case["identity"]) for case in json.loads(path.read_text())}, key=int), "scope": "All identities, including both tuning and assessment where applicable."}
    relative = "outputs/object_test/protocol.json"
    path = ROOT / relative
    exclusion_sources[relative] = {"sha256": sha(path), "identity_labels": sorted({str(case["unit"]) for case in json.loads(path.read_text())["cases"] if case["dataset"] == "celebahq"}, key=int), "scope": "Original HQ object-test identity labels; LaPa has no corresponding verified HQ identity labels."}
    smoke_path = ROOT / "outputs/reference_smoke/manifest.json"
    if smoke_path.is_file():
        exclusion_sources[str(smoke_path.relative_to(ROOT)).replace("\\", "/")] = {"sha256": sha(smoke_path), "identity_labels": sorted({str(case["identity_label"]) for case in json.loads(smoke_path.read_text())["images"]}, key=int), "scope": "Earlier reference engineering example."}
    exclusion_sources["explicit_engineering_example"] = {"identity_labels": ["916"], "scope": "Previously selected and inspected four-reference engineering example."}
    excluded = set().union(*(set(value["identity_labels"]) for value in exclusion_sources.values()))

    fingerprint_path = ROOT / "outputs/near_duplicate_audit/fingerprints.jsonl"
    fingerprints = {}
    with fingerprint_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["dataset"] == "celebahq":
                fingerprints[str(record["hq_id"])] = record
    # Exact cross-partition/group checks use the existing source-verified audit.
    protected = [record for record in fingerprints.values() if record["split"] != "val" or str(record["identity"]) in excluded]
    protected_file_hashes = {record["source_sha256"] for record in protected}
    protected_rgb_hashes = {record["decoded_rgb_sha256"] for record in protected}
    reviewed_paths = {str(Path(row["image_path"]).resolve()) for row in rows}
    review_path = ROOT / "research/near_duplicate_review_v2.json"
    review = json.loads(review_path.read_text())
    quarantined_paths = {str(Path(decision[key]).resolve()) for decision in review["decisions"] if decision["decision"] == "near_duplicate_photograph" for key in ["a_path", "b_path"]}
    if reviewed_paths & quarantined_paths:
        raise ValueError("Reviewed manifest unexpectedly retains a quarantined near-duplicate photograph.")

    cache = Path(json.loads((ROOT / "configs/local.json").read_text())["cache"])
    model_dir = cache / "reference_models/insightface/models/buffalo_l"
    model_sha = sha(model_dir / "det_10g.onnx")
    if model_sha != json.loads((ROOT / "research/reference_faces_provenance.json").read_text())["files"]["det_10g.onnx"]:
        raise ValueError("Face detector hash differs from recorded provenance.")
    signature = {
        "script_sha256": sha(__file__), "reviewed_manifest_sha256": sha(manifest_path),
        "fingerprints_sha256": sha(fingerprint_path), "review_sha256": sha(review_path),
        "exclusion_sources": exclusion_sources, "detector_sha256": model_sha,
        "seed": SEED, "requested_identity_count": COUNT, "references_per_target": 4,
        "detector_settings": {"provider": "CPUExecutionProvider", "det_size": [640, 640], "threshold": 0.5, "input_rgb_size": [512, 512]},
        "near_pair_hamming_exclusion_radius": 6,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    signature_path = OUT / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Diagnostic inputs changed. Create a new version instead of replacing this manifest.")
    write_json(signature_path, signature)

    eligibility = []
    for identity, identity_rows in sorted(groups.items(), key=lambda item: int(item[0])):
        reasons = []
        if identity in excluded:
            reasons.append("prior_experiment_or_engineering_identity")
        if len({row["source_sha256"] for row in identity_rows}) < 5:
            reasons.append("fewer_than_five_distinct_source_hashes")
        eligibility.append({"identity_label": identity, "reviewed_validation_images": len(identity_rows), "distinct_source_hashes": len({row["source_sha256"] for row in identity_rows}), "eligible_before_image_screening": not reasons, "exclusion_reasons": reasons})
    candidates = sorted((item["identity_label"] for item in eligibility if item["eligible_before_image_screening"]), key=lambda identity: ranked(identity, "identity"))
    audit = {
        "purpose": "Development-only diagnostic preparation; no generated-image performance consulted.",
        "selection_rule": "Only reviewed HQ validation rows. Exclude all prior pilot, tuning, assessment and original HQ test identity labels plus known engineering examples. Hash-rank identities and images with the frozen seed. Greedily retain five images per identity that pass exact hash, decoded-pixel hash, conservative pHash-distance (>6 from already retained diagnostic images), and exactly-one-face screening. First retained image is target; next four are references. Stop at twelve successful identities.",
        "eligibility": eligibility, "exclusion_sources": exclusion_sources, "attempts": [],
        "protected_exact_hash_comparison_images": len(protected),
        "photo_duplicate_screening": "Existing reviewed manifest quarantine + distinct original and decoded-pixel hashes + pHash distance >6 among all retained diagnostic photos. pHash rejection is conservative similarity screening, not proof of duplication or identity. Exact hashes are additionally checked against all existing HQ train/test audit records and excluded validation labels.",
    }
    write_json(OUT / "selection_audit.json", audit)
    cv2.setNumThreads(2)
    detector = FaceAnalysis(name=str(model_dir), allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    selected, retained_global = [], []
    global_file_hashes, global_rgb_hashes = set(), set()
    for rank, identity in enumerate(candidates, 1):
        attempt = {"rank": rank, "identity_label": identity, "screened_images": [], "selected": False}
        kept, kept_images = [], []
        for row in sorted(groups[identity], key=lambda item: ranked(item["hq_id"], "image")):
            source = Path(row["image_path"])
            file_hash = sha(source)
            if file_hash != row["source_sha256"]:
                raise ValueError(f"Source hash changed for HQ {row['hq_id']}.")
            fingerprint = fingerprints[row["hq_id"]]
            if fingerprint["source_sha256"] != file_hash:
                raise ValueError(f"Fingerprint source hash mismatch for HQ {row['hq_id']}.")
            with Image.open(source) as original:
                rgb = ImageOps.exif_transpose(original).convert("RGB")
                decoded_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                image = rgb.resize((512, 512), Image.Resampling.LANCZOS)
            if decoded_hash != fingerprint["decoded_rgb_sha256"]:
                raise ValueError(f"Decoded pixels differ from the audit for HQ {row['hq_id']}.")
            record = {"hq_id": row["hq_id"], "identity_label": identity, "source_path": str(source), "source_sha256": file_hash, "decoded_rgb_sha256": decoded_hash, "phash": fingerprint["phash"], "rejection_reasons": []}
            if file_hash in protected_file_hashes or decoded_hash in protected_rgb_hashes:
                record["rejection_reasons"].append("exact_photo_overlap_with_protected_prior_or_nonvalidation_image")
            if file_hash in global_file_hashes or file_hash in {item["source_sha256"] for item in kept}:
                record["rejection_reasons"].append("duplicate_source_hash_in_diagnostics")
            if decoded_hash in global_rgb_hashes or decoded_hash in {item["decoded_rgb_sha256"] for item in kept}:
                record["rejection_reasons"].append("duplicate_decoded_rgb_hash_in_diagnostics")
            near = [{"hq_id": item["hq_id"], "identity_label": item["identity_label"], "hamming_distance": (int(record["phash"], 16) ^ int(item["phash"], 16)).bit_count()} for item in retained_global + kept]
            record["near_pair_candidates"] = [pair for pair in near if pair["hamming_distance"] <= 6]
            if record["near_pair_candidates"]:
                record["rejection_reasons"].append("conservative_phash_near_pair_in_diagnostics")
            if not record["rejection_reasons"]:
                faces = detector.get(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                record["face_count"] = len(faces)
                record["detections"] = [{"score": float(face.det_score), "bbox": face.bbox.tolist(), "landmarks": face.kps.tolist()} for face in faces]
                if len(faces) != 1:
                    record["rejection_reasons"].append("face_count_is_not_one")
            else:
                record["face_detection_skipped"] = "Earlier deterministic duplicate screen rejected this photo."
            record["accepted"] = not record["rejection_reasons"]
            attempt["screened_images"].append(record)
            if record["accepted"]:
                kept.append(record)
                kept_images.append(image)
            if len(kept) == 5:
                break
        attempt["accepted_image_count"] = len(kept)
        attempt["selected"] = len(kept) == 5
        if not attempt["selected"]:
            attempt["rejection_reason"] = "fewer_than_five_images_passed_screening"
        else:
            folder = OUT / f"identity_{identity}"
            folder.mkdir(exist_ok=True)
            for index, (record, image) in enumerate(zip(kept, kept_images)):
                name = "target.png" if index == 0 else f"reference_{index}.png"
                image.save(folder / name)
                record.update(role="target" if index == 0 else f"reference_{index}", file=str((folder / name).relative_to(ROOT)).replace("\\", "/"), processed_sha256=sha(folder / name))
            eyes = np.asarray(kept[0]["detections"][0]["landmarks"][:2], dtype=float)
            delta = eyes[1] - eyes[0]
            distance = float(np.linalg.norm(delta))
            if distance < 1:
                raise ValueError("Degenerate target eye landmarks; no mask should be produced.")
            across, centre = delta / distance, eyes.mean(axis=0)
            down = np.array([-across[1], across[0]])
            polygon = [tuple((centre + x * distance * across + y * distance * down).tolist()) for x, y in [(-0.95, -0.35), (0.95, -0.35), (0.95, 0.35), (-0.95, 0.35)]]
            mask = Image.new("L", (512, 512), 0)
            ImageDraw.Draw(mask).polygon(polygon, fill=255)
            binary = np.asarray(mask) > 0
            assert all(binary[round(y), round(x)] for x, y in eyes)
            mask.save(folder / "mask.png")
            observed = Image.composite(Image.new("RGB", (512, 512), (90, 100, 110)), kept_images[0], mask)
            observed.save(folder / "observed.png")
            assert np.array_equal(np.asarray(observed)[~binary], np.asarray(kept_images[0])[~binary])
            selected.append({"case_id": f"identity_{identity}", "identity_label": identity, "source_split": "val", "images": kept, "observed": str((folder / "observed.png").relative_to(ROOT)).replace("\\", "/"), "observed_sha256": sha(folder / "observed.png"), "mask": str((folder / "mask.png").relative_to(ROOT)).replace("\\", "/"), "mask_sha256": sha(folder / "mask.png"), "mask_fraction": float(binary.mean()), "mask_geometry": {"kind": "synthetic eye-aligned rectangle", "polygon": polygon, "width_inter_eye_distances": 1.9, "height_inter_eye_distances": 0.7, "fill_rgb": [90, 100, 110], "uses_unobscured_target_landmarks_for_case_construction_only": True}})
            retained_global.extend(kept)
            global_file_hashes.update(item["source_sha256"] for item in kept)
            global_rgb_hashes.update(item["decoded_rgb_sha256"] for item in kept)
        audit["attempts"].append(attempt)
        write_json(OUT / "selection_audit.json", audit)
        print(f"Group {rank}/{len(candidates)} identity {identity}: {len(kept)} qualifying images; {len(selected)}/{COUNT} diagnostic identities selected", flush=True)
        if len(selected) == COUNT:
            break
    if len(selected) < COUNT:
        raise RuntimeError(f"Only {len(selected)} identities passed the frozen criteria; do not weaken them without versioning the protocol.")
    identities = {case["identity_label"] for case in selected}
    assert len(identities) == COUNT and not identities & excluded
    assert len(global_file_hashes) == len(global_rgb_hashes) == COUNT * 5
    assert all((int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count() > 6 for a, b in combinations(retained_global, 2))
    failures = Counter(reason for attempt in audit["attempts"] for image in attempt["screened_images"] for reason in image["rejection_reasons"])
    summary = {"validation_identity_labels": len(groups), "eligible_identity_labels_before_image_screening": len(candidates), "identity_groups_screened": len(audit["attempts"]), "failed_identity_groups": sum(not attempt["selected"] for attempt in audit["attempts"]), "photos_screened": sum(len(attempt["screened_images"]) for attempt in audit["attempts"]), "photo_rejection_reason_counts": dict(failures), "selected_identities": COUNT, "selected_photos": COUNT * 5, "selected_identity_labels": sorted(identities, key=int), "overlap_with_excluded_identity_labels": [], "source_hash_duplicates": 0, "decoded_rgb_duplicates": 0, "selected_phash_pairs_at_distance_le_6": 0, "all_selected_face_counts_equal_one": True}
    audit["summary"] = summary
    write_json(OUT / "selection_audit.json", audit)
    final_manifest = {
        "version": "reference_diagnostics_v1", "status": "development_only_not_fresh_final_test",
        "protocol_signature_sha256": sha(signature_path), "selection_audit_sha256": sha(OUT / "selection_audit.json"),
        "selection_rule": audit["selection_rule"], "summary": summary, "cases": selected,
        "usage": "For each case, pass observed plus mask and the reference_1..4 images to inference. Never supply target.png, unobscured target landmarks, or target pixels to the inpainting model. Target.png is retained only for later development comparisons. Reference-count or order studies must keep these chosen cases and all generated results, not select favourable outputs.",
        "limits": ["These are selected development cases, not a fresh final test and not a random population sample.", "Same-person grouping and exclusion use provided CelebA labels only. True identity independence and cross-dataset identity separation are not independently established.", "No generated output or measured inpainting performance influenced selection.", "Face-detection eligibility biases this diagnostic toward photographs with one detectable face.", "File and decoded-pixel hash checks plus conservative pHash screening do not prove absence of transformed duplicates or independent capture events.", "Cross-protected-set near-duplicate screening relies on the previous reviewed audit; the new conservative within-diagnostic screen is not an exhaustive crop/flip search.", "Synthetic masks use unobscured target eye landmarks during case construction. They do not represent naturally captured occlusions.", "Model pretraining overlap with these images is unresolved; validation-label exclusions do not establish pretraining independence.", "No model was trained or evaluated by this preparation script; no performance, novelty, or publication-readiness claim follows from these files."],
    }
    write_json(OUT / "manifest.json", final_manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
