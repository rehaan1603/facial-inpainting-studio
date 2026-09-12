"""Freeze a functional four-reference case using detection evidence only.

This is a validation-set demonstration, never a held-out performance estimate.
Original photographs and existing experiment protocols are not modified.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reference_examples"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_manifest = ROOT / "data/manifests/celebahq_reviewed_v2.csv"
    groups = defaultdict(list)
    with source_manifest.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "val":
                groups[row["identity"]].append(row)
    candidates = sorted(
        (identity for identity, rows in groups.items() if len(rows) >= 5),
        key=lambda identity: hashlib.sha256(("reference-smoke-17:" + identity).encode()).hexdigest(),
    )
    cache = Path(json.loads((ROOT / "configs/local.json").read_text())["cache"])
    model_dir = cache / "reference_models/insightface/models/buffalo_l"
    provenance = json.loads((ROOT / "research/reference_faces_provenance.json").read_text())
    detector_sha = sha(model_dir / "det_10g.onnx")
    assert detector_sha == provenance["files"]["det_10g.onnx"]
    detector = FaceAnalysis(name=str(model_dir), allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    audit = {
        "purpose": "Deterministic functional demonstration of four-reference inpainting, not a scientific performance estimate.",
        "selection_rule": "Validation identity labels with at least five reviewed images, ordered by SHA256('reference-smoke-17:' + identity). For each label, inspect the first five numeric HQ IDs. Select the first group with five different source hashes and exactly one detected face in every 512px image. Do not select a later image within a rejected group.",
        "selection_evidence": "Source hashes and CPU face detection only. No generated images or inpainting performance are consulted.",
        "excluded_before_detection": {"identity_groups_with_fewer_than_five_images": sum(len(rows) < 5 for rows in groups.values())},
        "candidate_group_count": len(candidates),
        "manifest_sha256": sha(source_manifest),
        "detector": {"model": "InsightFace buffalo_l det_10g.onnx", "sha256": detector_sha, "provider": "CPUExecutionProvider", "det_size": [640, 640], "det_threshold": 0.5},
        "attempts": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    selected = None
    for rank, identity in enumerate(candidates, 1):
        rows = sorted(groups[identity], key=lambda row: int(row["hq_id"]))[:5]
        attempt = {"rank": rank, "identity_label": identity, "images": [], "rejections": []}
        images, faces_by_image, hashes = [], [], []
        for index, row in enumerate(rows):
            path = Path(row["image_path"])
            digest = sha(path)
            if digest != row["source_sha256"]:
                raise ValueError(f"Source hash changed for HQ {row['hq_id']}; refusing to select.")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            faces = detector.get(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
            entry = {"role": "target" if index == 0 else f"reference_{index}", "hq_id": row["hq_id"], "identity_label": identity, "source_path": str(path), "source_sha256": digest, "face_count": len(faces), "detections": [{"score": float(face.det_score), "bbox": face.bbox.tolist(), "landmarks": face.kps.tolist()} for face in faces]}
            attempt["images"].append(entry)
            if len(faces) != 1:
                attempt["rejections"].append(f"{entry['role']} HQ {row['hq_id']} has {len(faces)} detected faces; exactly one is required.")
            images.append(image)
            faces_by_image.append(faces)
            hashes.append(digest)
        if len(set(hashes)) != 5:
            attempt["rejections"].append("The five images do not have distinct source hashes.")
        attempt["selected"] = not attempt["rejections"]
        audit["attempts"].append(attempt)
        (OUT / "selection_audit.json").write_text(json.dumps(audit, indent=2))
        print(f"Group {rank}, identity label {identity}: {'selected' if attempt['selected'] else '; '.join(attempt['rejections'])}", flush=True)
        if attempt["selected"]:
            selected = (identity, images, faces_by_image, attempt)
            break
    if selected is None:
        raise RuntimeError("No candidate met the frozen detection criteria; see selection_audit.json.")
    identity, images, faces_by_image, attempt = selected
    out = OUT / f"identity_{identity}"
    out.mkdir(exist_ok=True)
    for index, image in enumerate(images):
        name = "target.png" if index == 0 else f"reference_{index}.png"
        image.save(out / name)
        attempt["images"][index].update({"file": name, "processed_sha256": sha(out / name)})
    eyes = np.asarray(faces_by_image[0][0].kps[:2], dtype=float)
    direction = eyes[1] - eyes[0]
    distance = float(np.linalg.norm(direction))
    if distance < 1:
        raise ValueError("Degenerate eye landmarks; do not create a misleading mask.")
    horizontal = direction / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    centre = eyes.mean(axis=0)
    polygon = [centre + x * distance * horizontal + y * distance * vertical for x, y in [(-0.95, -0.35), (0.95, -0.35), (0.95, 0.35), (-0.95, 0.35)]]
    polygon = [tuple(point.tolist()) for point in polygon]
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).polygon(polygon, fill=255)
    mask_array = np.asarray(mask)
    assert all(mask_array[round(y), round(x)] == 255 for x, y in eyes)
    assert set(np.unique(mask_array)) == {0, 255}
    mask.save(out / "mask.png")
    observed = Image.composite(Image.new("RGB", (512, 512), (90, 100, 110)), images[0], mask)
    observed.save(out / "observed.png")
    assert np.array_equal(np.asarray(observed)[mask_array == 0], np.asarray(images[0])[mask_array == 0])
    manifest = {
        "identity_label": identity,
        "split": "val",
        "selection_rule": audit["selection_rule"],
        "selection_audit": "../selection_audit.json",
        "images": attempt["images"],
        "mask": {"file": "mask.png", "white_means": "region to reconstruct", "source": "target face detector's two eye landmarks", "eye_landmarks": eyes.tolist(), "polygon": polygon, "geometry": "Eye-aligned rectangle; width 1.9 times inter-eye distance and height 0.7 times inter-eye distance, centred on the two eyes.", "fraction": float((mask_array > 0).mean()), "sha256": sha(out / "mask.png")},
        "observed": {"file": "observed.png", "fill_rgb": [90, 100, 110], "sha256": sha(out / "observed.png"), "outside_mask_exactly_preserved": True},
        "limits": ["Same-person grouping follows supplied CelebA identity labels; it is not independently verified identity ground truth.", "Distinct source hashes establish distinct files, not independent capture events or absence of near duplicates.", "This selected validation example is for functional debugging only. It is not a held-out test or evidence of reconstruction quality, novelty, or publication readiness.", "Occlusion is synthetic and aligned using unobscured target landmarks. Target.png is withheld from inference and retained only as a debugging comparison.", "Reference model pretraining overlap with these photographs is unresolved."],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    audit["selected_directory"] = str(out)
    (OUT / "selection_audit.json").write_text(json.dumps(audit, indent=2))
    print(out, flush=True)


if __name__ == "__main__":
    main()
