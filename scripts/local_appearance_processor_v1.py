"""Route aggregate reference appearance by target query, retaining joint attention.

The native per-image mask path recomputes softmax separately for each reference.
This wrapper instead delegates the untouched joint-reference processor. It keeps
the full output inside the supplied hole and the appearance-off output outside.
No target originals, landmarks, RGB transfer or learned parameters are used.
"""
import math
import torch
from torch.nn import functional as F
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0


class LocalAppearanceProcessor(IPAdapterAttnProcessor2_0):
    def __init__(self, delegate, target_mask):
        # Reuse the original weight modules; do not allocate another adapter.
        torch.nn.Module.__init__(self)
        if not isinstance(delegate, IPAdapterAttnProcessor2_0):
            raise TypeError('This study requires the existing Torch2 IP-Adapter processor')
        if len(delegate.scale) != 2:
            raise ValueError('Expected identity and appearance branches')
        mask = torch.as_tensor(target_mask, dtype=torch.bool).detach().cpu()
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
            raise ValueError('Expected a square target mask')
        self.delegate = delegate
        self.register_buffer('target_mask', mask, persistent=False)
        self.query_grids = {}

    @property
    def scale(self):
        return self.delegate.scale

    @scale.setter
    def scale(self, value):
        self.delegate.scale = value

    @property
    def num_tokens(self):
        return self.delegate.num_tokens

    @num_tokens.setter
    def num_tokens(self, value):
        self.delegate.num_tokens = value

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, ip_adapter_masks=None, **kwargs):
        if ip_adapter_masks is not None:
            raise ValueError('Do not combine native per-reference masks with aggregate routing')
        arguments = dict(encoder_hidden_states=encoder_hidden_states,
                         attention_mask=attention_mask, temb=temb, **kwargs)
        scales = self.delegate.scale
        if not isinstance(scales[1], (float, int)):
            raise ValueError('The fixed appearance branch uses one scalar scale')
        # Exact original code path for intervention-off / full-mask controls.
        full = self.delegate(attn, hidden_states, **arguments)
        if scales[1] == 0 or bool(self.target_mask.all()):
            return full
        try:
            self.delegate.scale = [scales[0], 0.]
            base = self.delegate(attn, hidden_states, **arguments)
        finally:
            self.delegate.scale = scales
        if not bool(self.target_mask.any()):
            return base
        if full.ndim == 3:
            side = math.isqrt(full.shape[1])
            if side * side != full.shape[1]:
                raise ValueError('Only square query grids are supported in this study')
            mask = F.interpolate(self.target_mask.float()[None, None], (side, side), mode='nearest')
            self.query_grids[str(side)] = dict(side=side, selected_queries=int(mask.sum()), total_queries=side*side)
            mask = mask.reshape(1, side * side, 1).bool().to(full.device)
        elif full.ndim == 4:
            mask = F.interpolate(self.target_mask.float()[None, None], full.shape[-2:], mode='nearest').bool().to(full.device)
        else:
            raise ValueError('Unexpected attention output rank')
        return torch.where(mask, full, base)


def install(pipe, mask):
    count = 0
    processors = {}
    for name, processor in pipe.unet.attn_processors.items():
        if isinstance(processor, IPAdapterAttnProcessor2_0):
            processors[name] = LocalAppearanceProcessor(processor, mask)
            count += 1
        else:
            processors[name] = processor
    if not count:
        raise ValueError('No supported IP-Adapter processors were found')
    pipe.unet.set_attn_processor(processors)
    return count


def set_mask(pipe, mask):
    for processor in pipe.unet.attn_processors.values():
        if isinstance(processor, LocalAppearanceProcessor):
            replacement = torch.as_tensor(mask, dtype=torch.bool, device=processor.target_mask.device)
            if replacement.shape != processor.target_mask.shape:
                raise ValueError('Target mask shape changed')
            processor.target_mask.copy_(replacement)
