"""Experimental attention adapter; separate from production and frozen studies."""
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_model():
    cache = Path(json.loads((ROOT / 'configs/local.json').read_text())['cache'])
    source = cache / 'refldm_source_v1'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == 'af6690c19fdc6421802fd7996510bcfef259bfd1'
    weights = cache / 'refldm_weights_v1'
    for entry in json.loads((ROOT / 'research/refldm_downloads_v1.json').read_text())['files']:
        assert sha(weights / entry['name']) == entry['sha256']
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    config = OmegaConf.load(source / 'configs/refldm.yaml')
    config.model.params.first_stage_config.params.ckpt_path = str(weights / 'vqgan.ckpt')
    for key in ['ckpt_path', 'perceptual_loss_config']:
        config.model.params.pop(key, None)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(weights / 'refldm.ckpt', map_location='cpu', weights_only=True), strict=True)
    if model.use_ema:
        model.model_ema.copy_to(model.model)
    return model.requires_grad_(False).eval()


class ReferenceBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv1d(32, 16, 1), torch.nn.SiLU(), torch.nn.Conv1d(16, 1, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.enabled = True
        self.drop_references = False

    def forward(self, values):
        return 4 * torch.tanh(self.net(values.float()))


def install_adapter(model, adapter):
    from ldm import cache_kv
    from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy
    count = 0
    for module in model.model.diffusion_model.modules():
        if not isinstance(module, QKVAttentionLegacy):
            continue
        original = module.forward

        def forward(self, qkv, original=original):
            if cache_kv.mode != 'use' or (not adapter.enabled and not adapter.drop_references):
                return original(qkv)
            batch, width, length = qkv.shape
            channels = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(batch * self.n_heads, channels * 3, length).split(channels, dim=1)
            biases = [torch.zeros_like(q[:, :1, :])]
            if not adapter.drop_references:
                refs = cache_kv.v[id(self)]
                biases += [adapter(value).to(q.dtype) for value in refs]
                k = torch.cat([k] + cache_kv.k[id(self)], -1)
                v = torch.cat([v] + refs, -1)
            scale = 1 / math.sqrt(math.sqrt(channels))
            logits = torch.einsum('bct,bcs->bts', q * scale, k * scale)
            weights = torch.softmax((logits + torch.cat(biases, -1)).float(), dim=-1).to(q.dtype)
            return torch.einsum('bts,bcs->bct', weights, v).reshape(batch, -1, length)

        module.forward = types.MethodType(forward, module)
        count += 1
    if count == 0:
        raise RuntimeError('No compatible attention layers found')
    return count


def set_cache(model, refs):
    from ldm import cache_kv
    cache_kv.clear_cache()
    cache_kv.mode = 'save'
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for ref in refs:
            image = torch.cat([ref, torch.zeros_like(ref)], dim=1)
            model.model.diffusion_model(image, torch.zeros(1, device='cuda', dtype=torch.long), is_ref=True)
    cache_kv.mode = 'use'


def predict(model, noisy, lq, timestep):
    return model.model.diffusion_model(torch.cat([noisy, lq], dim=1), timestep)
