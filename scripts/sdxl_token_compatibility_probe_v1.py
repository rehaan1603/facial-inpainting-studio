"""Synthetic direct-SDXL backward feasibility; no photographs or checkpoints.

The shared scorer changes appearance-token attention only in a supplied query
mask. Tokens are learned IP-Adapter tokens, not anatomical correspondences.
The executable requires an already-saved protocol and never downloads assets.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import time
import urllib.error
import urllib.request
from importlib.metadata import version
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from local_appearance_processor_v1 import LocalAppearanceProcessor

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['query_key_cosine', 'normalized_rms_difference',
            'normalized_channel_mean_difference', 'normalized_channel_std_difference',
            'log1p_query_norm', 'log1p_key_norm', 'log1p_norm_difference',
            'key_consensus_cosine', 'timestep_fraction', 'visible_target_fraction']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


class CompatibilityScorer(nn.Module):
    """385 registered parameters, shared across processors/heads/references."""
    def __init__(self, timestep=500, visible_fraction=.9375):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.timestep_fraction = float(timestep) / 999.
        self.visible_fraction = float(visible_fraction)
        self.enabled = True

    def forward(self, query, key):
        # Reductions/matmul avoid a [batch,head,query,key,channel] pair tensor.
        # No inputs are detached: later adapted queries retain their graph.
        with torch.autocast(device_type=query.device.type, enabled=False):
            q, k = query.float(), key.float()
            qnorm, knorm = q.norm(dim=-1, keepdim=True), k.norm(dim=-1, keepdim=True)
            qn, kn = q / qnorm.clamp_min(1e-6), k / knorm.clamp_min(1e-6)
            cosine = torch.matmul(qn, kn.transpose(-1, -2)).clamp(-1, 1)
            shape = cosine.shape
            expand = lambda x: x.expand(shape)
            qmean, kmean = qn.mean(-1, keepdim=True), kn.mean(-1).unsqueeze(-2)
            qstd, kstd = qn.std(-1, unbiased=False, keepdim=True), kn.std(-1, unbiased=False).unsqueeze(-2)
            consensus = F.normalize(kn.mean(-2, keepdim=True), dim=-1)
            key_consensus = (kn * consensus).sum(-1).unsqueeze(-2)
            values = [cosine, ((2 - 2 * cosine).clamp_min(1e-8) / q.shape[-1]).sqrt(),
                      (qmean - kmean).abs(), (qstd - kstd).abs(),
                      expand(qnorm.log1p()), expand(knorm.squeeze(-1).unsqueeze(-2).log1p()),
                      (qnorm - knorm.squeeze(-1).unsqueeze(-2)).abs().log1p(),
                      expand(key_consensus), torch.full_like(cosine, self.timestep_fraction),
                      torch.full_like(cosine, self.visible_fraction)]
            features = torch.stack(values, dim=-1)
            return (.5 * torch.tanh(self.net(features).squeeze(-1))).to(query.dtype)


class TokenCompatibilityProcessor(IPAdapterAttnProcessor2_0):
    """Frozen joint-reference attention plus bounded learned appearance bias.

    Text and FaceID SDPA are unchanged. Appearance queries outside the mask
    contribute zero, matching the local-appearance study algebraically.
    """
    def __init__(self, delegate, scorer, target_mask, chunk_size=64):
        nn.Module.__init__(self)
        if not isinstance(delegate, IPAdapterAttnProcessor2_0) or len(delegate.scale) != 2:
            raise TypeError('Expected two-branch Torch2 IP-Adapter processor')
        self.delegate = delegate
        self.scorer = scorer
        self.register_buffer('target_mask', torch.as_tensor(target_mask, dtype=torch.bool), persistent=False)
        self.chunk_size = int(chunk_size)

    @property
    def scale(self): return self.delegate.scale
    @scale.setter
    def scale(self, value): self.delegate.scale = value
    @property
    def num_tokens(self): return self.delegate.num_tokens
    @num_tokens.setter
    def num_tokens(self, value): self.delegate.num_tokens = value

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, ip_adapter_masks=None, **kwargs):
        if ip_adapter_masks is not None:
            raise ValueError('Native per-reference mask attention is a different protocol')
        if not isinstance(encoder_hidden_states, tuple) or len(encoder_hidden_states[1]) != 2:
            raise ValueError('Expected text and separate FaceID/appearance token banks')
        text, banks = encoder_hidden_states
        residual, ndim = hidden_states, hidden_states.ndim
        if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
        if ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        batch, length, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if attn.norm_cross: text = attn.norm_encoder_hidden_states(text)
        query, key, value = attn.to_q(hidden_states), attn.to_k(text), attn.to_v(text)
        head_dim = key.shape[-1] // attn.heads
        reshape = lambda x: x.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        query, key, value = reshape(query), reshape(key), reshape(value)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, text.shape[1], batch)
            attention_mask = attention_mask.view(batch, attn.heads, -1, attention_mask.shape[-1])
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0., is_causal=False)
        side = math.isqrt(length)
        if side * side != length: raise ValueError('Probe requires square target query grids')
        query_mask = F.interpolate(self.target_mask.float()[None, None], (side, side), mode='nearest')
        query_mask = query_mask.reshape(1, 1, length, 1).to(query.device, dtype=query.dtype)
        for index, (bank, scale, to_k, to_v) in enumerate(zip(banks, self.scale, self.delegate.to_k_ip, self.delegate.to_v_ip)):
            if not isinstance(scale, (float, int)): raise ValueError('Fixed scalar branch scales required')
            if scale == 0: continue
            # Flatten all four reference token sets jointly, as original processor.
            ip_key, ip_value = reshape(to_k(bank)), reshape(to_v(bank))
            if index == 0 or not self.scorer.enabled:
                extra = F.scaled_dot_product_attention(query, ip_key, ip_value, dropout_p=0., is_causal=False)
            else:
                chunks = []
                for start in range(0, length, self.chunk_size):
                    stop = min(start + self.chunk_size, length)
                    q = query[:, :, start:stop]
                    # Checkpoint MLP feature/hidden activations; additive SDPA bias
                    # remains bounded and joint across references. No zero bypass.
                    if torch.is_grad_enabled():
                        bias = checkpoint(self.scorer, q, ip_key, use_reentrant=False)
                    else: bias = self.scorer(q, ip_key)
                    bias = bias * query_mask[:, :, start:stop, 0].unsqueeze(-1)
                    chunks.append(F.scaled_dot_product_attention(q, ip_key, ip_value, attn_mask=bias, dropout_p=0., is_causal=False))
                extra = torch.cat(chunks, dim=-2)
            if index == 1: extra = extra * query_mask
            result = result + scale * extra
        result = result.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
        result = attn.to_out[1](attn.to_out[0](result))
        if ndim == 4: result = result.transpose(-1, -2).reshape(batch, channels, height, width)
        if attn.residual_connection: result = result + residual
        return result / attn.rescale_output_factor


def install(unet, scorer, mask, chunk_size=64):
    processors, count = {}, 0
    for name, processor in unet.attn_processors.items():
        if isinstance(processor, IPAdapterAttnProcessor2_0):
            processors[name] = TokenCompatibilityProcessor(processor, scorer, mask, chunk_size)
            count += 1
        else: processors[name] = processor
    if not count: raise ValueError('No supported IP-Adapter attention processors')
    unet.set_attn_processor(processors)
    return count


def ensure_studio_idle():
    for port in (8765, 8766):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/session', timeout=2) as response:
                busy = json.load(response).get('busy')
            if busy:
                raise RuntimeError(f'Studio port {port} is busy; do not compete for GPU memory')
        except (urllib.error.URLError, TimeoutError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=[512, 1024], required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite an existing probe receipt')
    if not args.protocol.is_file(): raise ValueError('Save and freeze the protocol before the probe')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before model loading, including failures; only this invocation updates it.
    receipt = dict(status='running', scope='Synthetic single epsilon-loss backward feasibility; no dataset training or quality evidence',
                   resolution=args.resolution, latent_size=args.resolution // 8, seed=20261010,
                   protocol_path=str(args.protocol.resolve()), protocol_sha256=sha(args.protocol), script_sha256=sha(__file__),
                   feature_order=FEATURES, trainable_parameters=385, bias_bound=.5, query_chunk=64,
                   precision='fp16 frozen weights, bfloat16 autocast, float32 scorer', checkpointing='UNet and non-reentrant scorer chunks',
                   neutral_tolerance=1e-3, references=4, tokens_per_reference=[16,16],
                   branch_scales=[.8,.2], timestep=500, optimizer='AdamW lr=0.0001, one disposable update',
                   exclusions=['dataset pixels', 'VAE', 'image-space/identity losses', 'CFG sampling', 'training checkpoint', 'quality claims'],
                   feature_gradients='Query/key feature graph retained; no detachment',
                   note='Partly correlated features; zero initialization initially gives zero first-layer gradients. The shared scalar offset has no effective attention gradient at neutral initialization.')
    with args.output.open('x', encoding='utf-8') as f: json.dump(receipt, f, indent=2)
    start = time.monotonic()
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for the explicit GPU feasibility probe')
        ensure_studio_idle()
        from diffusers import UNet2DConditionModel
        from safetensors.torch import load_file
        import diffusers.models.attention_processor as attention_source
        cache = Path(json.loads((ROOT/'configs/local.json').read_text())['cache'])
        sd = cache/'osor_models/sdxl'
        assets = []
        expected = json.loads((ROOT/'research/osor_downloads.json').read_text())
        for name in ['unet/config.json', 'unet/diffusion_pytorch_model.fp16.safetensors']:
            record = next(x for x in expected if x['repository']=='diffusers/stable-diffusion-xl-1.0-inpainting-0.1' and x['filename']==name)
            assert record['revision']=='115134f363124c53c7d878647567d04daf26e41e'
            path = sd/name
            assert sha(path)==record['sha256'], 'Pinned SDXL asset mismatch'
            assets.append(dict(path=str(path),sha256=record['sha256'],revision=record['revision']))
        portrait = cache/'reference_models/ip-adapter-faceid-portrait_sdxl.bin'
        baseline = json.loads((ROOT/'outputs/focused_eye_factorial_v1/config.json').read_text())
        assert sha(portrait)==baseline['adapter_sha256']
        assets.append(dict(path=str(portrait),sha256=baseline['adapter_sha256']))
        appearance = cache/'eye_appearance_models_v1/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors'
        appearance_receipt = json.loads((cache/'eye_appearance_models_v1/receipt.json').read_text())
        assert appearance_receipt['revision']=='018e402774aeeddd60609b4ecdb7e298259dc729'
        app_pin = next(x for x in appearance_receipt['files'] if Path(x['path']).name==appearance.name)
        assert sha(appearance)==app_pin['sha256']
        assets.append(dict(path=str(appearance),sha256=app_pin['sha256'],revision=appearance_receipt['revision']))
        receipt.update(assets=assets, gpu=torch.cuda.get_device_name(), versions={x:version(x) for x in ['torch','diffusers','transformers','safetensors']},
                       local_routing_source_sha256=sha(ROOT/'scripts/local_appearance_processor_v1.py'),
                       attention_source_sha256=sha(attention_source.__file__), unet_source_sha256=sha(inspect.getfile(UNet2DConditionModel)))
        torch.set_num_threads(4)
        torch.manual_seed(receipt['seed'])
        unet = UNet2DConditionModel.from_pretrained(str(sd), subfolder='unet', variant='fp16', torch_dtype=torch.float16, local_files_only=True)
        portrait_state = torch.load(portrait,map_location='cpu',weights_only=True)
        tensors = load_file(str(appearance),device='cpu')
        appearance_state = {prefix:{k[len(prefix)+1:]:v for k,v in tensors.items() if k.startswith(prefix+'.')} for prefix in ['image_proj','ip_adapter']}
        unet._load_ip_adapter_weights([portrait_state,appearance_state])
        embedded_loras = unet._load_ip_adapter_loras([portrait_state,appearance_state])
        if embedded_loras:
            raise ValueError('Unexpected embedded FaceID LoRA; direct-loader contract must be audited separately')
        projections = unet.encoder_hid_proj.image_projection_layers
        actual_tokens = [int(projections[0].num_tokens), int(projections[1].latents.shape[1])]
        if actual_tokens != [16,16]: raise ValueError('Unexpected reference projection token layout')
        receipt['actual_projection_tokens_per_reference'] = actual_tokens
        receipt['embedded_loras'] = 0
        del portrait_state,appearance_state,tensors
        unet.requires_grad_(False).train()
        receipt['loaded_frozen_parameters'] = sum(p.numel() for p in unet.parameters())
        assert unet.config.in_channels==9
        for processor in unet.attn_processors.values():
            if isinstance(processor,IPAdapterAttnProcessor2_0): processor.num_tokens=[16,16];processor.scale=[.8,.2]
        unet.enable_gradient_checkpointing()
        ensure_studio_idle()
        unet.cuda()
        side = args.resolution // 8
        mask = torch.zeros(side,side,device='cuda',dtype=torch.bool)
        mask[side*3//8:side*5//8,side*3//8:side*5//8] = True
        scorer = CompatibilityScorer(500,float((~mask).float().mean())).cuda()
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        sample = torch.randn(1,9,side,side,device='cuda',dtype=torch.float16)
        sample[:,4] = mask.to(sample.dtype)
        sample[:,5:] *= (~mask)[None,None]
        text = torch.randn(1,77,2048,device='cuda',dtype=torch.float16)
        identity = F.normalize(torch.randn(1,4,512,device='cuda',dtype=torch.float32),dim=-1).half()
        appearance_features = torch.randn(1,4,257,1280,device='cuda',dtype=torch.float16)
        added = dict(text_embeds=torch.randn(1,1280,device='cuda',dtype=torch.float16),
                     time_ids=torch.tensor([[args.resolution,args.resolution,0,0,args.resolution,args.resolution]],device='cuda',dtype=torch.float16),
                     image_embeds=[identity,appearance_features])
        timestep = torch.tensor([500],device='cuda')
        def forward():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return unet(sample,timestep,encoder_hidden_states=text,added_cond_kwargs=added,return_dict=False)[0]
        # Full-backbone neutral control uses the preserved routing helper rather
        # than another mode of the new processor under test.
        original_processors = dict(unet.attn_processors)
        local_processors = {name: LocalAppearanceProcessor(p, mask) if isinstance(p, IPAdapterAttnProcessor2_0) else p
                            for name, p in original_processors.items()}
        unet.set_attn_processor(local_processors)
        ensure_studio_idle()
        with torch.no_grad(): baseline_pred=forward().detach()
        unet.set_attn_processor(original_processors.copy())
        receipt['attention_processors'] = install(unet,scorer,mask)
        receipt['actual_trainable_parameters'] = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        assert receipt['actual_trainable_parameters']==385
        with torch.no_grad(): neutral_pred=forward().detach()
        neutral_error=float((baseline_pred.float()-neutral_pred.float()).abs().max())
        receipt['neutral_max_absolute_error']=neutral_error
        if neutral_error>receipt['neutral_tolerance']: raise RuntimeError('Neutral synthetic prediction exceeded fixed tolerance')
        del baseline_pred,neutral_pred
        params=list(scorer.parameters());before=[p.detach().clone() for p in params]
        optimizer=torch.optim.AdamW(params,lr=1e-4)
        target_noise=torch.randn(1,4,side,side,device='cuda',dtype=torch.float32)
        backward_start=time.monotonic()
        ensure_studio_idle()
        prediction=forward()
        error=(prediction.float()-target_noise).square()
        hole=mask.float()[None,None]
        loss=.25*error.mean()+.75*(error*hole).sum()/(4*hole.sum())
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite probe loss')
        loss.backward()
        gradients=[p.grad for p in params if p.grad is not None]
        if len(gradients)!=len(params) or not all(torch.isfinite(g).all() for g in gradients): raise RuntimeError('Missing/nonfinite scorer gradients')
        norm=float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
        if norm<=0: raise RuntimeError('Zero scorer gradient')
        frozen_gradients=any(p.grad is not None for p in unet.parameters() if not p.requires_grad)
        if frozen_gradients: raise RuntimeError('Frozen weights acquired gradients')
        optimizer.step();torch.cuda.synchronize()
        changed=any(not torch.equal(old,p) for old,p in zip(before,params))
        if not changed: raise RuntimeError('Temporary scorer did not update')
        receipt.update(status='passed',loss=float(loss.detach()),gradient_norm=norm,parameter_update=changed,
                       frozen_weights_have_gradients=False,backward_step_seconds=time.monotonic()-backward_start,
                       gradient_per_parameter=[dict(shape=list(p.shape),present=p.grad is not None,norm=float(p.grad.float().norm()) if p.grad is not None else None) for p in params])
    except Exception as exc:
        receipt.update(status='failed',error=f'{type(exc).__name__}: {exc}',out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError))
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start)
        if torch.cuda.is_available(): receipt.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(receipt,indent=2,allow_nan=False),flush=True)
    if receipt['status']!='passed': raise SystemExit(1)


if __name__ == '__main__': main()
