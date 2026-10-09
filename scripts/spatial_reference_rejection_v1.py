"""Research-only oracle spatial reference-token exclusion. No learned parameters."""
import hashlib
import math
import types
import torch


def region_box(policy, identity):
    if policy == 'neutral':
        return None
    if policy == 'oracle_region':
        return (.25, .25, .75, .75)
    if policy != 'random_corner':
        raise ValueError(policy)
    index = hashlib.sha256(('spatial-rejection-v1:' + str(identity)).encode()).digest()[0] % 4
    x, y = (index % 2) * .5, (index // 2) * .5
    return (x, y, x + .5, y + .5)


def token_mask(length, box, device):
    side = math.isqrt(length)
    if side * side != length or side % 4:
        raise ValueError('Expected square spatial tokens divisible by four')
    mask = torch.zeros(side, side, dtype=torch.bool, device=device)
    x0, y0, x1, y1 = box
    mask[int(y0*side):int(y1*side), int(x0*side):int(x1*side)] = True
    assert int(mask.sum()) == length // 4
    return mask.flatten()


class SpatialRejection:
    def __init__(self):
        self.box = None
        self.calls = 0
        self.token_counts = set()


def install(model, intervention):
    from ldm import cache_kv
    from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy
    count = 0
    for module in model.model.diffusion_model.modules():
        if not isinstance(module, QKVAttentionLegacy):
            continue
        original = module.forward

        def forward(self, qkv, original=original):
            if cache_kv.mode != 'use' or intervention.box is None:
                return original(qkv)
            batch, width, length = qkv.shape
            if batch != 2:
                raise ValueError('This experiment requires author CFG null-first/conditional-second batch')
            channels = width // (3*self.n_heads)
            q, k, v = qkv.reshape(batch*self.n_heads, 3*channels, length).split(channels, dim=1)
            keys, values = cache_kv.k[id(self)], cache_kv.v[id(self)]
            assert len(keys) == len(values) == 4
            mask = token_mask(keys[0].shape[-1], intervention.box, q.device)
            k = torch.cat([k]+keys, -1)
            v = torch.cat([v]+values, -1)
            scale = 1/math.sqrt(math.sqrt(channels))
            logits = torch.einsum('bct,bcs->bts', q*scale, k*scale)
            excluded = torch.zeros(logits.shape[-1], dtype=torch.bool, device=q.device)
            excluded[length:length+mask.numel()] = mask
            # Keep null branch, self tokens and other references byte-identical.
            logits[self.n_heads:] = logits[self.n_heads:].masked_fill(excluded[None,None,:], -torch.inf)
            weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
            intervention.calls += 1
            intervention.token_counts.add((mask.numel(), int(mask.sum())))
            return torch.einsum('bts,bcs->bct', weights, v).reshape(batch, -1, length)

        module.forward = types.MethodType(forward, module)
        count += 1
    if count != 16:
        raise ValueError(f'Expected 16 attention layers, found {count}')
    return count
