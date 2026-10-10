"""Prepare a new, private, CPU-only masked SDXL pilot cache; never train a model.

Read the accompanying data preflight before execution. No ML imports or image
decoding happen unless --prepare is supplied. All output belongs to the new cache.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Set these before any ML package is imported. This script has no CUDA mode.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/reference_intervention_pilot_v1/manifest.json"
PREFLIGHT = ROOT / "research/MASKED_SDXL_COMPATIBILITY_DATA_PREFLIGHT_V1.md"
OUT = ROOT / "outputs/masked_sdxl_compatibility_pilot_v1/cache"
MANIFEST_SHA = "f01f52bae44001bc9c23d36c0b4c5b48cd8e60e8f19cb1ba717be085143f117e"
TRAIN_IDS = ["8570", "6792", "4410", "2250", "3930", "8173", "2083", "7506",
             "4625", "38", "4164", "674", "8660", "476", "1281", "8748"]
FORBIDDEN = {"263", "6681", "10149", "7400"}
SDXL_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
SDXL_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
FACEID_REVISION = "43907e6f44d079bf1a9102d9a6e56aef7a219bae"
APPEARANCE_REVISION = "018e402774aeeddd60609b4ecdb7e298259dc729"
PROMPT = ("A realistic portrait photograph of a person, natural facial features, "
          "consistent lighting, detailed skin.")
ANNOTATION_NAMES = ["l_eye", "r_eye", "l_brow", "r_brow"]
SOURCE_PINS = {
    "configs/local.json": "059b977d654e2cbedd1c54307acee6988bbb7e9b0b6f8b3fbcd2825a5d1277de",
    "scripts/reference_inpaint.py": "6fa4fe9c7742c8a1ec69d1586cc98a07573b1b85e5bebf0ad4c899b75a782daf",
    "scripts/run_focused_eye_appearance_v1.py": "df5b44c99893c3512072ec33be92db5c1fc55b9939f3fc9f0767abc34b53fef4",
    "scripts/benchmark_v2.py": "6eb4f4f35db971e28c219bd1de4bac629ac38d44ad80729945763dec9e6395d5",
    "research/osor_downloads.json": "d226ddcab433603fecce08467ef52ce304a4f8ddab313d6df475eea10df9ef1b",
    "research/reference_adapter_provenance.json": "066a4446733b1c336a3c9785fe0ae2c5f0cdd941434cd8c4d71cf1f00285a358",
    "research/reference_faces_provenance.json": "9cbc23137333833e675b9d7d3c142ee895418503bedf62cdf91aace7f34b69be",
}
APPEARANCE_RECEIPT_SHA = "7283ad25980157b21f0b47a3e404966f5f57cf1799159a052562c5ae7b4277f8"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(path: Path, expected: str, label: str) -> dict:
    require(path.is_file(), f"Missing {label}: {path}")
    actual = sha(path)
    require(actual == expected, f"SHA256 mismatch for {label}: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def atomic_json(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def metadata_preflight() -> tuple[dict, dict, dict]:
    """Verify metadata and allowed source bytes only; do not decode any image."""
    require(not OUT.exists(), f"Cache already exists; refusing overwrite/resume: {OUT}")
    require(PREFLIGHT.is_file(), "The data preflight document must exist before preparation")
    pins = {name: verify(ROOT / name, expected, name) for name, expected in SOURCE_PINS.items()}
    pins["manifest"] = verify(MANIFEST, MANIFEST_SHA, "frozen 16-case source manifest")
    pins["preflight"] = verify(PREFLIGHT, sha(PREFLIGHT), "preflight document")
    pins["preparer"] = verify(Path(__file__), sha(Path(__file__)), "preparation script")
    local = json.loads((ROOT / "configs/local.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = ROOT / "data/manifests/celebahq_reviewed_v2.csv"
    fingerprints = ROOT / "outputs/near_duplicate_audit/fingerprints.jsonl"
    pins["reviewed_csv"] = verify(source, manifest["source_sha256"], "reviewed CSV")
    pins["fingerprints"] = verify(fingerprints, manifest["fingerprints_sha256"], "frozen fingerprints")
    with source.open(encoding="utf-8", newline="") as stream:
        reviewed = {row["hq_id"]: row for row in csv.DictReader(stream)}
    cases = [case for case in manifest["cases"] if case["role"] == "train"]
    require([str(case["identity"]) for case in cases] == TRAIN_IDS,
            "Training identity count/order differs from the frozen 16 cases")
    excluded = set(map(str, manifest["excluded_identities"])) | FORBIDDEN
    excluded |= {str(case["identity"]) for case in manifest["cases"] if case["role"] != "train"}
    require(not set(TRAIN_IDS) & excluded, "Training identities overlap excluded/holdout identities")
    hq_images = Path(local["hq"]) / "CelebA-HQ-img"
    seen_ids, seen_hashes = set(), set()
    for case in cases:
        identity = str(case["identity"])
        require(len(case["references"]) == 4, f"Identity {identity} needs exactly four real references")
        for role, row in [("target", case["target"]), *[("reference", r) for r in case["references"]]]:
            require(str(row["identity"]) == identity and row["split"] == "train",
                    f"Invalid identity/partition for {identity}/{role}")
            index = str(row["hq_id"])
            path = Path(row["image_path"])
            require(index in reviewed, f"HQ ID {index} absent from reviewed CSV")
            original = reviewed[index]
            require(all(str(row[k]) == str(original[k]) for k in
                        ["hq_id", "original_file", "identity", "split", "source_sha256"]),
                    f"Source record metadata changed for HQ ID {index}")
            require(path.resolve() == Path(original["image_path"]).resolve(),
                    f"Source record path changed for HQ ID {index}")
            require(inside(path, hq_images) and path.resolve() == (hq_images / f"{index}.jpg").resolve(),
                    f"Source path is outside the permitted HQ image directory: {path}")
            require(index not in seen_ids and row["source_sha256"] not in seen_hashes,
                    f"Target/reference reuse or duplicate source bytes at HQ ID {index}")
            verify(path, row["source_sha256"], f"{identity}/{role}/{index}")
            seen_ids.add(index)
            seen_hashes.add(row["source_sha256"])
    require(len(seen_ids) == 80, "The cache must use exactly 16 targets and 64 separate references")
    return local, manifest, {"sources": pins, "cases": cases, "excluded": sorted(excluded, key=int)}


def asset_preflight(cache: Path) -> dict:
    """Verify every model file to be consumed; load no weight tensors here."""
    sd = cache / "osor_models/sdxl"
    records = json.loads((ROOT / "research/osor_downloads.json").read_text(encoding="utf-8"))
    selected = [r for r in records if r["repository"] == SDXL_REPO]
    require(all(r["revision"] == SDXL_REVISION for r in selected), "SDXL revision mismatch")
    consumed = {"model_index.json", "scheduler/scheduler_config.json", "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors", "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors", "text_encoder_2/config.json",
                "text_encoder_2/model.fp16.safetensors"}
    consumed |= {f"{folder}/{name}" for folder in ["tokenizer", "tokenizer_2"] for name in
                 ["merges.txt", "vocab.json", "special_tokens_map.json", "tokenizer_config.json"]}
    by_name = {r["filename"]: r for r in selected}
    require(consumed <= by_name.keys(), "Missing SDXL asset provenance")
    assets = []
    for name in sorted(consumed):
        record = by_name[name]
        assets.append({**verify(sd / name, record["sha256"], name), "repository": SDXL_REPO,
                       "revision": SDXL_REVISION, "filename": name})
    for folder in ["text_encoder", "text_encoder_2"]:
        name = folder + "/model.safetensors"
        expected = by_name[folder + "/model.fp16.safetensors"]["sha256"]
        assets.append({**verify(sd / name, expected, "baseline default text-weight alias"),
                       "repository": SDXL_REPO, "revision": SDXL_REVISION, "filename": name,
                       "alias_of": folder + "/model.fp16.safetensors"})
    faceid = json.loads((ROOT / "research/reference_adapter_provenance.json").read_text(encoding="utf-8"))
    require(faceid["revision"] == FACEID_REVISION, "Portrait revision mismatch")
    assets.append({**verify(cache / "reference_models" / faceid["file"], faceid["sha256"], "FaceID Portrait"),
                   "repository": faceid["repository"], "revision": FACEID_REVISION, "loaded": False})
    faces = json.loads((ROOT / "research/reference_faces_provenance.json").read_text(encoding="utf-8"))
    face_dir = cache / "reference_models/insightface/models/buffalo_l"
    require({p.name for p in face_dir.glob("*.onnx")} == set(faces["files"]),
            "Unexpected unpinned ONNX file in the buffalo_l directory")
    require(not (face_dir / "manifest.json").exists(), "Unpinned InsightFace model-package manifest")
    for name, expected in faces["files"].items():
        assets.append({**verify(face_dir / name, expected, "buffalo_l " + name), "source": faces["source"]})
    app_root = cache / "eye_appearance_models_v1"
    app_receipt = verify(app_root / "receipt.json", APPEARANCE_RECEIPT_SHA, "appearance provenance")
    appearance = json.loads((app_root / "receipt.json").read_text(encoding="utf-8"))
    require(appearance["repository"] == "h94/IP-Adapter" and
            appearance["revision"] == APPEARANCE_REVISION, "Appearance revision mismatch")
    require({r["file"] for r in appearance["files"]} == {
        "models/image_encoder/config.json", "models/image_encoder/model.safetensors",
        "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"}, "Unexpected appearance asset set")
    for record in appearance["files"]:
        path = app_root / record["file"]
        require(path.resolve() == Path(record["path"]).resolve(), "Appearance asset path mismatch")
        assets.append({**verify(path, record["sha256"], record["file"]),
                       "repository": appearance["repository"], "revision": APPEARANCE_REVISION,
                       "loaded": record["file"].startswith("models/image_encoder/")})
    scheduler_config = json.loads((sd / "scheduler/scheduler_config.json").read_text(encoding="utf-8"))
    require(scheduler_config.get("prediction_type") == "epsilon", "The pinned SDXL scheduler must predict epsilon")
    return {"assets": assets, "appearance_receipt": app_receipt, "sdxl": str(sd),
            "scheduler_config": scheduler_config,
            "face_dir": str(face_dir), "appearance_encoder": str(app_root / "models/image_encoder")}


def implementation_pin(obj) -> dict:
    path = Path(inspect.getfile(obj))
    return {"path": str(path), "sha256": sha(path), "name": obj.__name__}


def load_rgb(path: Path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def semantic_function(np, Image):
    """Reuse only the pinned function; importing benchmark_v2 would import evaluators."""
    tree = ast.parse((ROOT / "scripts/benchmark_v2.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "semantic_mask"]
    require(len(functions) == 1, "Could not isolate the existing semantic_mask")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"np": np, "Image": Image}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / "scripts/benchmark_v2.py"), "exec"), namespace)
    return namespace["semantic_mask"]


def assert_tensor(torch, tensor, shape, dtype, name: str) -> None:
    require(tuple(tensor.shape) == tuple(shape), f"Unexpected shape for {name}: {tuple(tensor.shape)}")
    require(tensor.device.type == "cpu" and tensor.dtype == dtype, f"Unexpected device/dtype for {name}")
    require(bool(torch.isfinite(tensor).all()), f"Nonfinite {name}")


def save_tensors(torch, path: Path, tensors: dict) -> dict:
    require(not path.exists(), f"Refusing tensor overwrite: {path}")
    for name, tensor in tensors.items():
        require(tensor.device.type == "cpu" and bool(torch.isfinite(tensor).all()),
                f"Invalid cached tensor {name}")
    temporary = path.with_name(path.name + ".partial")
    torch.save(tensors, temporary)
    temporary.replace(path)
    return {"file": path.name, "path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size,
            "tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype), "device": "cpu"}
                        for name, t in tensors.items()}}


def prepare() -> dict:
    start = time.monotonic()
    local, manifest, checked = metadata_preflight()
    assets = asset_preflight(Path(local["cache"]))
    # Imports follow all role/source/model-hash guards and explicit CLI selection.
    import cv2
    import numpy as np
    import psutil
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLInpaintPipeline
    from diffusers.image_processor import VaeImageProcessor
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from transformers import CLIPVisionModelWithProjection

    torch.set_num_threads(4)
    torch.set_default_device("cpu")
    cv2.setNumThreads(1)
    for package, expected in {"diffusers": "0.40.0", "transformers": "5.17.0", "insightface": "2.0"}.items():
        require(version(package) == expected, f"Version changed for {package}; review a new preflight")
    OUT.mkdir(parents=True, exist_ok=False)
    receipt_path = OUT / "receipt.json"
    receipt = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Local 16-identity training-input cache only; no training/evaluation/inference",
               "manifest_sha256": MANIFEST_SHA, "sources": checked["sources"],
               "asset_provenance": assets, "train_identities": TRAIN_IDS,
               "excluded_identities": checked["excluded"], "expected_counts": {
                   "records": 16, "targets": 16, "references": 64, "decoded_photographs": 80,
                   "vae_encodes": 32, "evaluation_photographs_decoded": 0,
                   "evaluation_annotations_decoded": 0, "training_updates": 0},
               "versions": {name: version(name) for name in ["torch", "diffusers", "transformers", "tokenizers",
                            "safetensors", "insightface", "onnxruntime", "opencv-python", "pillow", "numpy"]},
               "python": sys.version, "executable": sys.executable,
               "device": "cpu", "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
               "threads": {"torch": 4, "opencv": 1}, "stage_seconds": {}, "records": [],
               "protocol": {"resolution": 512, "mask": "benchmark_v2.semantic_mask eyes 512 padding8",
                    "masked_image": "normalized clean target * (mask < 0.5)", "latent_mask_mode": "nearest",
                    "vae_dtype": "float32", "vae_posterior": "sample with independent seeded CPU generators",
                    "faceid_dtype": "float32; cast to diffusion dtype at consumption", "faceid_order": "negative, positive",
                    "appearance_dtype": "float32; pre-cast features from CPU float32 vision model",
                    "appearance_order": "negative, positive", "text_dtype": "float16 on CPU",
                    "historical_replay_claim": False},
               "implementations": {name: implementation_pin(obj) for name, obj in {
                   "encode_prompt": StableDiffusionXLInpaintPipeline.encode_prompt, "vae": AutoencoderKL,
                   "vae_image_processor": VaeImageProcessor,
                   "text_encoder": CLIPTextModel, "projected_text_encoder": CLIPTextModelWithProjection,
                   "tokenizer": CLIPTokenizer, "vision_encoder": CLIPVisionModelWithProjection,
                   "image_processor": CLIPImageProcessor, "face_analysis": FaceAnalysis,
                   "norm_crop": face_align.norm_crop}.items()}}
    process = psutil.Process()

    def progress(stage: str):
        memory = process.memory_info()
        receipt["current_stage"] = stage
        receipt["process_rss_bytes"] = memory.rss
        receipt["process_peak_working_set_bytes"] = getattr(memory, "peak_wset", None)
        receipt["elapsed_seconds"] = time.monotonic() - start
        atomic_json(receipt_path, receipt)
        print(stage, flush=True)

    try:
        progress("text_conditioning")
        stage_start = time.monotonic()
        sd = Path(assets["sdxl"])
        tokenizer = CLIPTokenizer.from_pretrained(str(sd / "tokenizer"), local_files_only=True)
        tokenizer_2 = CLIPTokenizer.from_pretrained(str(sd / "tokenizer_2"), local_files_only=True)
        encoder = CLIPTextModel.from_pretrained(str(sd / "text_encoder"), local_files_only=True,
                                                torch_dtype=torch.float16, use_safetensors=True).cpu().eval()
        encoder_2 = CLIPTextModelWithProjection.from_pretrained(str(sd / "text_encoder_2"), local_files_only=True,
                                                torch_dtype=torch.float16, use_safetensors=True).cpu().eval()
        for model in [encoder, encoder_2]:
            model.requires_grad_(False)
            require(all(p.device.type == "cpu" and p.dtype == torch.float16 for p in model.parameters()),
                    "Text model must use CPU float16 weights")
        require(tokenizer.model_max_length == tokenizer_2.model_max_length == 77, "Unexpected token length")
        pipeline = StableDiffusionXLInpaintPipeline(vae=None, unet=None, text_encoder=encoder,
            text_encoder_2=encoder_2, tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            scheduler=DDIMScheduler.from_pretrained(str(sd / "scheduler"), local_files_only=True),
            add_watermarker=False, force_zeros_for_empty_prompt=True, requires_aesthetics_score=False)
        with torch.inference_mode():
            positive, _, pooled, _ = pipeline.encode_prompt(prompt=PROMPT, prompt_2=None, device=torch.device("cpu"),
                num_images_per_prompt=1, do_classifier_free_guidance=False, clip_skip=None, lora_scale=None)
        assert_tensor(torch, positive, [1, 77, 2048], torch.float16, "positive prompt")
        assert_tensor(torch, pooled, [1, 1280], torch.float16, "pooled prompt")
        time_ids = torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=torch.float16, device="cpu")
        receipt["common_condition"] = {**save_tensors(torch, OUT / "text_conditioning.pt", {
            "prompt_embeds": positive.contiguous(), "pooled_prompt_embeds": pooled.contiguous(),
            "add_time_ids": time_ids}), "prompt": PROMPT, "method": "existing Diffusers encode_prompt",
            "num_images_per_prompt": 1, "clip_skip": None, "unet_loaded": False,
            "tokenizer_max_lengths": [77, 77]}
        del pipeline, encoder, encoder_2, tokenizer, tokenizer_2, positive, pooled, time_ids
        gc.collect()
        receipt["stage_seconds"]["text_conditioning"] = time.monotonic() - stage_start

        progress("target_and_masked_vae_latents")
        stage_start = time.monotonic()
        semantic_mask = semantic_function(np, Image)
        image_processor = VaeImageProcessor(vae_scale_factor=8, do_resize=False)
        vae = AutoencoderKL.from_pretrained(str(sd), subfolder="vae", variant="fp16",
                    torch_dtype=torch.float32, local_files_only=True, use_safetensors=True).float().cpu().eval()
        vae.requires_grad_(False)
        vae.disable_tiling()
        vae.disable_slicing()
        require(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in vae.parameters()),
                "VAE must use CPU float32 weights")
        require(float(vae.config.scaling_factor) == 0.13025, "Unexpected VAE scaling factor")
        receipt["vae"] = {"scaling_factor": float(vae.config.scaling_factor),
                          "tiling": False, "slicing": False, "sampling_device": "cpu"}
        latents_by_id, metadata_by_id = {}, {}
        for case in checked["cases"]:
            identity, target = str(case["identity"]), case["target"]
            index = int(target["hq_id"])
            annotation_info = []
            for name in ANNOTATION_NAMES:
                path = Path(local["hq"]) / "CelebAMask-HQ-mask-anno" / str(index // 2000) / f"{index:05d}_{name}.png"
                annotation_info.append({"component": name, "path": str(path), "present": path.is_file(),
                                        "sha256": sha(path) if path.is_file() else None})
            binary = semantic_mask(Path(local["hq"]), index, "eyes", 512)
            require(binary is not None and tuple(binary.shape) == (512, 512), f"No valid eye mask for {identity}")
            coverage = float(binary.mean())
            require(0.01 < coverage < 0.65, f"Extreme eye-mask coverage for {identity}: {coverage}")
            verify(Path(target["image_path"]), target["source_sha256"], "target immediately before decode")
            rgb = load_rgb(Path(target["image_path"])).resize((512, 512), Image.Resampling.LANCZOS)
            clean = image_processor.preprocess(rgb, height=512, width=512).contiguous()
            assert_tensor(torch, clean, [1, 3, 512, 512], torch.float32, "normalized training target")
            mask = torch.from_numpy(binary.astype(np.float32))[None, None]
            masked = clean * (mask < 0.5)
            latent_mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode="nearest")
            require(bool(((latent_mask == 0) | (latent_mask == 1)).all()), "Latent mask must stay binary")
            seeds = {"target": int(manifest["seed"]) + 2 * index, "masked_image": int(manifest["seed"]) + 2 * index + 1}
            samples, posterior_info = {}, {}
            for name, pixels in [("target", clean), ("masked_image", masked)]:
                with torch.inference_mode():
                    posterior = vae.encode(pixels).latent_dist
                    assert_tensor(torch, posterior.mean, [1, 4, 64, 64], torch.float32, name + " posterior mean")
                    assert_tensor(torch, posterior.logvar, [1, 4, 64, 64], torch.float32, name + " posterior logvar")
                    sample = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(seeds[name]))
                    sample = (sample * float(vae.config.scaling_factor)).contiguous()
                assert_tensor(torch, sample, [1, 4, 64, 64], torch.float32, name + " latents")
                samples["target_latent" if name == "target" else "masked_latent"] = sample
                posterior_info[name] = {"dtype": "float32", "device": "cpu", "method": "sample",
                                        "seed": seeds[name], "shape": [1, 4, 64, 64]}
                del posterior, sample
            yy, xx = np.where(binary)
            latents_by_id[identity] = {**samples, "mask": latent_mask.contiguous()}
            metadata_by_id[identity] = {"identity": identity, "role": "train", "target": target,
                "references": case["references"], "annotations": annotation_info, "posterior": posterior_info,
                "visible_fraction": 1.0 - coverage, "latent_visible_fraction": 1.0 - float(latent_mask.mean()),
                "mask": {"coverage": coverage, "bbox_xyxy_exclusive": [int(xx.min()), int(yy.min()),
                    int(xx.max()) + 1, int(yy.max()) + 1], "binary_sha256": hashlib.sha256(binary.tobytes()).hexdigest(),
                    "family": "eyes", "resolution": 512, "padding": 8, "latent_interpolation": "nearest"}}
            del clean, masked, mask, latent_mask, rgb, binary, samples
            progress(f"vae_complete_{identity}")
        del vae, image_processor
        gc.collect()
        receipt["stage_seconds"]["target_and_masked_vae_latents"] = time.monotonic() - stage_start

        progress("reference_faceid_and_appearance")
        stage_start = time.monotonic()
        analyser = FaceAnalysis(name=assets["face_dir"], allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        analyser.prepare(ctx_id=-1, det_size=(640, 640))
        for model in analyser.models.values():
            require(model.session.get_providers() == ["CPUExecutionProvider"], "Non-CPU face-analysis provider")
        vision = CLIPVisionModelWithProjection.from_pretrained(assets["appearance_encoder"], local_files_only=True,
                                                               torch_dtype=torch.float32).float().cpu().eval()
        vision.requires_grad_(False)
        require(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in vision.parameters()),
                "Vision encoder must use CPU float32 weights")
        processor = CLIPImageProcessor()
        receipt["image_processor"] = {"class": type(processor).__name__, "config": processor.to_dict(),
                                      "historical_backend_replay_claim": False}
        for case in checked["cases"]:
            identity = str(case["identity"])
            crops, faceids, detections = [], [], []
            receipt["current_identity"] = identity
            progress(f"references_start_{identity}")
            for position, reference in enumerate(case["references"]):
                verify(Path(reference["image_path"]), reference["source_sha256"], "reference immediately before decode")
                rgb = np.asarray(load_rgb(Path(reference["image_path"])))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                faces = analyser.get(bgr)
                require(len(faces) == 1, f"Identity {identity} reference {position} must have exactly one face; found {len(faces)}")
                embedding, landmarks = faces[0].normed_embedding, faces[0].kps
                require(embedding is not None and embedding.shape == (512,) and bool(np.isfinite(embedding).all()),
                        f"Invalid FaceID embedding at {identity}/{position}")
                require(landmarks is not None and landmarks.shape == (5, 2) and bool(np.isfinite(landmarks).all()),
                        f"Invalid five-point landmarks at {identity}/{position}")
                require(np.isfinite(float(faces[0].det_score)), f"Nonfinite detection score at {identity}/{position}")
                crop = face_align.norm_crop(bgr, landmark=landmarks, image_size=224)[:, :, ::-1].copy()
                crops.append(Image.fromarray(crop))
                faceids.append(embedding.copy())
                detections.append({"position": position, "hq_id": str(reference["hq_id"]),
                    "source_sha256": reference["source_sha256"], "face_count": 1,
                    "detection_score": float(faces[0].det_score), "landmarks_5pt": landmarks.tolist(),
                    "aligned_rgb_bytes_sha256": hashlib.sha256(crop.tobytes()).hexdigest()})
            faceid_positive = torch.from_numpy(np.stack(faceids)).float()[None]
            faceid_bank = torch.cat([torch.zeros_like(faceid_positive), faceid_positive]).contiguous()
            pixels = processor(images=crops, return_tensors="pt").pixel_values.float().cpu()
            assert_tensor(torch, pixels, [4, 3, 224, 224], torch.float32, "reference CLIP pixels")
            with torch.inference_mode():
                positive = vision.vision_model(pixels, output_hidden_states=True).hidden_states[-2]
                negative = vision.vision_model(torch.zeros_like(pixels), output_hidden_states=True).hidden_states[-2]
            appearance_bank = torch.stack([negative, positive]).float().cpu().contiguous()
            assert_tensor(torch, faceid_bank, [2, 4, 512], torch.float32, "FaceID bank")
            assert_tensor(torch, appearance_bank, [2, 4, 257, 1280], torch.float32, "appearance bank")
            require(bool((faceid_bank[0] == 0).all()), "FaceID unconditional bank must be zero")
            tensors = {**latents_by_id.pop(identity), "identity": faceid_bank, "appearance": appearance_bank}
            record = {**metadata_by_id[identity], "reference_detections": detections,
                "faceid": {"encoder": "buffalo_l w600k_r50", "normalization": "InsightFace normed_embedding",
                           "provider": "CPUExecutionProvider", "det_size": [640, 640], "references_averaged": False},
                "appearance": {"layer": "penultimate hidden state", "alignment": "InsightFace five-point norm_crop 224",
                               "unconditional": "vision_model applied to zeros_like(processed_pixels)",
                               "references_averaged": False},
                **save_tensors(torch, OUT / f"identity_{identity}.pt", tensors)}
            record["tensor_path"] = record["path"]
            receipt["records"].append(record)
            del tensors, crops, faceids, detections, faceid_positive, faceid_bank, pixels, positive, negative, appearance_bank
            gc.collect()
            progress(f"cache_complete_{identity}")
        del vision, analyser, processor
        gc.collect()
        receipt["stage_seconds"]["reference_faceid_and_appearance"] = time.monotonic() - stage_start
        require([r["identity"] for r in receipt["records"]] == TRAIN_IDS and not latents_by_id,
                "Incomplete or reordered cache")
        for record in [receipt["common_condition"], *receipt["records"]]:
            verify(OUT / record["file"], record["sha256"], "final tensor integrity")
        receipt["status"] = "complete"
        receipt["counts"] = dict(receipt["expected_counts"])
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["cache_tensor_bytes"] = sum(r["bytes"] for r in [receipt["common_condition"], *receipt["records"]])
        receipt.pop("current_identity", None)
        progress("complete")
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error),
                              "stage": receipt.get("current_stage"), "identity": receipt.get("current_identity")}
        receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["elapsed_seconds"] = time.monotonic() - start
        atomic_json(receipt_path, receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Execute the reviewed CPU-only cache preparation")
    args = parser.parse_args()
    if not args.prepare:
        print("No preparation executed. Read research/MASKED_SDXL_COMPATIBILITY_DATA_PREFLIGHT_V1.md; use --prepare explicitly.")
        return
    receipt = prepare()
    print(json.dumps({"status": receipt["status"], "cache": str(OUT), "records": len(receipt["records"]),
                      "elapsed_seconds": receipt["elapsed_seconds"], "tensor_bytes": receipt["cache_tensor_bytes"]}))


if __name__ == "__main__":
    main()
