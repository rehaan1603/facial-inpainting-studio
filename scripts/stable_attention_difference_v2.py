"""First-order SDPA residual with local cancellation of native-input gradients.

Both branches always execute. No zero-weight branch, parameter-dependent
detach, or surrogate derivative. Recompute first derivatives independently and
subtract before returning to the native graph, avoiding accumulation roundoff.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)


class AttentionDifference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dq, dv):
        ctx.save_for_backward(q, k, v, dq, dv)
        with torch.autocast(device_type=q.device.type, enabled=False):
            return sdpa(q+dq, k, v+dv)-sdpa(q, k, v)

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        q, k, v, dq, dv = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(device_type=q.device.type, enabled=False):
            qa = (q+dq).detach().requires_grad_(True)
            ka = k.detach().requires_grad_(True)
            va = (v+dv).detach().requires_grad_(True)
            qb = q.detach().requires_grad_(True)
            kb = k.detach().requires_grad_(True)
            vb = v.detach().requires_grad_(True)
            ga = torch.autograd.grad(sdpa(qa, ka, va), (qa, ka, va), output_gradient)
            gb = torch.autograd.grad(sdpa(qb, kb, vb), (qb, kb, vb), output_gradient)
        acc = torch.float64 if q.dtype == torch.float64 else torch.float32
        native = tuple((a.to(acc)-b.to(acc)).to(a.dtype) for a, b in zip(ga, gb))
        return (*native, ga[0], ga[2])


attention_difference = AttentionDifference.apply
