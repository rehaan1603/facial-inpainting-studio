import json
from pathlib import Path
import torch

torch.manual_seed(20260910)
assert torch.cuda.is_available(), 'CUDA is required; use the project .venv'
x = torch.randn(2, 3, 256, 256, device='cuda', requires_grad=True)
model = torch.nn.Conv2d(3, 16, 3).cuda()
loss = model(x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
report = {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'device': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
          'forward_backward': 'passed', 'loss': loss.item()}
root = Path(__file__).resolve().parents[1]
(root / 'research/gpu_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
