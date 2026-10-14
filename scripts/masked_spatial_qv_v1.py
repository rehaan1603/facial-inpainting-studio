"""Audited rank-4 spatial Q/V residual; no model loading or execution on import.

Only the late SDXL self-attention layer identified by HOOK is supported. The
native processor result is preserved, with a separately calculated attention
difference added only at nearest-resized original hole queries. This module
does not claim downstream image locality or GPU numerical neutrality.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


HOOK = 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor'
WIDTH = 640
HEADS = 10
RANK = 4
SEED = 20261012
SCALING = 1.0
PARAMETER_COUNT = 10240


class SpatialQVAdapter(nn.Module):
    """Four fp32 parameters, isolated CPU initialization, no extra gate/bias.

    Device moves are supported, including movement by model CPU offload. Dtype
    conversions of a containing UNet leave the master adapter weights fp32 and
    preserve their original values. Forward copies across devices, if needed,
    remain differentiable and do not replace registered Parameters.
    """

    def __init__(self):
        super().__init__()
        generator = torch.Generator(device='cpu').manual_seed(SEED)
        self.q_a = nn.Parameter(torch.empty(RANK, WIDTH, dtype=torch.float32))
        self.q_b = nn.Parameter(torch.zeros(WIDTH, RANK, dtype=torch.float32))
        self.v_a = nn.Parameter(torch.empty(RANK, WIDTH, dtype=torch.float32))
        self.v_b = nn.Parameter(torch.zeros(WIDTH, RANK, dtype=torch.float32))
        # Exactly nn.Linear.reset_parameters' weight initialization, without
        # constructing a Linear that would first consume the global RNG.
        nn.init.kaiming_uniform_(self.q_a, a=math.sqrt(5), generator=generator)
        nn.init.kaiming_uniform_(self.v_a, a=math.sqrt(5), generator=generator)

    def _apply(self, fn, recurse=True):
        def keep_fp32(tensor):
            converted = fn(tensor)
            if tensor.is_floating_point():
                # Convert directly from the original fp32 tensor, never from
                # an already rounded half-precision intermediate.
                return tensor.to(device=converted.device, dtype=torch.float32)
            return converted
        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, query_hidden, all_hidden):
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('Adapter master parameters must remain float32')
        if query_hidden.shape[-1] != WIDTH or all_hidden.shape[-1] != WIDTH:
            raise ValueError('Adapter requires 640-channel hidden states')
        if query_hidden.device != all_hidden.device:
            raise ValueError('Q/V inputs must use one execution device')
        device = query_hidden.device
        with torch.autocast(device_type=device.type, enabled=False):
            q = F.linear(F.linear(query_hidden.float(), self.q_a.to(device)), self.q_b.to(device))
            v = F.linear(F.linear(all_hidden.float(), self.v_a.to(device)), self.v_b.to(device))
        return SCALING * q, SCALING * v


def _validate_attention(attn):
    """Fail closed on settings that change the audited attention algebra."""
    if attn.heads != HEADS or attn.inner_dim != WIDTH or attn.inner_kv_dim != WIDTH:
        raise ValueError('Audited attention must have 10 heads and 640 Q/K/V channels')
    for name in ('to_q', 'to_k', 'to_v'):
        projection = getattr(attn, name, None)
        if type(projection) is not nn.Linear or tuple(projection.weight.shape) != (WIDTH, WIDTH):
            raise ValueError(f'Unsupported {name}: expected frozen native 640x640 Linear')
        if projection.bias is not None:
            raise ValueError('Audited Q/K/V projections have no bias')
    if (attn.to_out is None or len(attn.to_out) != 2
            or type(attn.to_out[0]) is not nn.Linear
            or tuple(attn.to_out[0].weight.shape) != (WIDTH, WIDTH)
            or type(attn.to_out[1]) is not nn.Dropout or attn.to_out[1].p != 0):
        raise ValueError('Audited output must be 640x640 Linear and zero dropout')
    if attn.to_out[0].bias is None:
        raise ValueError('Audited output projection has a native bias')
    if any(getattr(attn, name, None) is not None for name in
           ('spatial_norm', 'group_norm', 'norm_q', 'norm_k', 'norm_cross',
            'added_kv_proj_dim', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out')):
        raise ValueError('Additional normalization/projections are unsupported')
    if (attn.is_cross_attention or attn.only_cross_attention or attn.residual_connection
            or attn.rescale_output_factor != 1.0 or attn.upcast_attention or attn.upcast_softmax
            or attn.fused_projections or attn.is_causal or attn.pre_only
            or attn.context_pre_only is not None or not attn.scale_qk
            or attn.scale != (WIDTH // HEADS) ** -0.5):
        raise ValueError('Unsupported setting at audited SDXL self-attention')
    for projection in (attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]):
        if any(p.requires_grad for p in projection.parameters()):
            raise ValueError('Native attention maps must remain frozen')


class SpatialQVProcessor(nn.Module):
    """Preserve the original delegate; add a masked Q/V attention difference.

    ``target_mask`` is a nonpersistent buffer; call ``update_mask`` for each
    original example. An empty nearest-resized hole returns the delegate result.
    There is intentionally no enabled flag or zero-weight short circuit.
    """

    def __init__(self, delegate, adapter, target_mask):
        super().__init__()
        if type(delegate) is not AttnProcessor2_0:
            raise TypeError('Requires the unchanged native AttnProcessor2_0 delegate')
        if type(adapter) is not SpatialQVAdapter:
            raise TypeError('Requires the fixed SpatialQVAdapter')
        self.delegate = delegate
        self.adapter = adapter
        self.register_buffer('target_mask', None, persistent=False)
        self.update_mask(target_mask)

    def update_mask(self, mask):
        mask = torch.as_tensor(mask)
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or mask.shape[0] < 1:
            raise ValueError('Original hole mask must be a nonempty square 2D array')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Original hole mask must contain only binary values')
        self.target_mask = mask.detach().to(dtype=torch.bool).clone()

    def query_indices(self, length, device):
        side = math.isqrt(length)
        if side * side != length or length < 1:
            raise ValueError('Audited self-attention requires a square spatial query grid')
        mask = F.interpolate(self.target_mask.to(device=device, dtype=torch.float32)[None, None],
                             size=(side, side), mode='nearest')
        return mask.reshape(-1).nonzero(as_tuple=False).flatten()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        # Diffusers introspects this call signature. Accept then explicitly
        # reject unsupported controls rather than silently using other algebra.
        if encoder_hidden_states is not None or attention_mask is not None or temb is not None:
            raise ValueError('Only unmasked audited self-attention inputs are supported')
        if args or kwargs:
            raise ValueError(f'Unsupported attention processor arguments: {tuple(kwargs)}')
        _validate_attention(attn)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != WIDTH:
            raise ValueError('Expected [batch, square spatial length, 640] hidden states')
        indices = self.query_indices(hidden_states.shape[1], hidden_states.device)
        base = self.delegate(attn, hidden_states)
        if indices.numel() == 0:
            return base
        batch = hidden_states.shape[0]
        query_hidden = hidden_states.index_select(1, indices)
        native_q = attn.to_q(query_hidden)
        native_k = attn.to_k(hidden_states)
        native_v = attn.to_v(hidden_states)
        delta_q, delta_v = self.adapter(query_hidden, hidden_states)
        adapted_q = native_q + delta_q.to(native_q.dtype)
        adapted_v = native_v + delta_v.to(native_v.dtype)

        def heads(tensor):
            return tensor.view(batch, -1, HEADS, WIDTH // HEADS).transpose(1, 2)

        key = heads(native_k)
        # Both calls use exactly the same query/key/value shapes and layouts.
        baseline = F.scaled_dot_product_attention(heads(native_q), key, heads(native_v),
                                                 dropout_p=0., is_causal=False)
        adapted = F.scaled_dot_product_attention(heads(adapted_q), key, heads(adapted_v),
                                                dropout_p=0., is_causal=False)
        difference = (adapted - baseline).transpose(1, 2).reshape(batch, indices.numel(), WIDTH)
        # The delegate already includes the output bias. Apply only its frozen
        # linear map to the difference; zero dropout and rescale=1 are guarded.
        delta = F.linear(difference.to(native_q.dtype), attn.to_out[0].weight, bias=None)
        # index_copy preserves every visible value exactly (including signed
        # zero), instead of doing an addition across the entire visible grid.
        return base.index_copy(1, indices, base.index_select(1, indices) + delta.to(base.dtype))


def install(unet, adapter, mask):
    """Install one processor, retaining all existing compatibility processors.

    Install after loading/freezing the native weights and existing scorer. The
    returned processor owns the adapter as a registered child module, so its
    10,240 parameters appear in UNet.parameters alongside the original scorer.
    """
    processors = dict(unet.attn_processors)
    if HOOK not in processors:
        raise ValueError('Audited SDXL attention hook is missing')
    attention = unet.get_submodule(HOOK.removesuffix('.processor'))
    _validate_attention(attention)
    if sum(p.numel() for p in adapter.parameters()) != PARAMETER_COUNT:
        raise ValueError('Adapter parameter count changed')
    if any(p.dtype != torch.float32 or not p.requires_grad for p in adapter.parameters()):
        raise ValueError('All 10,240 adapter parameters must be trainable float32')
    processor = SpatialQVProcessor(processors[HOOK], adapter, mask)
    # Mutating only the target Attention avoids replacing/discarding any of the
    # 70 preexisting cross-attention compatibility processor objects.
    attention.set_processor(processor)
    after = unet.attn_processors
    if any(after[name] is not previous for name, previous in processors.items() if name != HOOK):
        raise RuntimeError('Installing Q/V unexpectedly changed another attention processor')
    return processor
