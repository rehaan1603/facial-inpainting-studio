"""Inference-only adapter for the published RefFaceInpainting baseline.

Author source/weights remain in the local cache. No target truth, training losses,
discriminators, optimizer state or dataset labels are model inputs here.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def rgb_tensor(image, size, device):
    image = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1


def conservative_mask(mask, size=256):
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Expected an explicit binary two-dimensional missing mask')
    tensor = torch.from_numpy(mask.copy())[None, None].float()
    return F.adaptive_max_pool2d(tensor, (size, size))


class RefFaceBaseline:
    def __init__(self, source, parser_source, weights, device='cuda'):
        self.device = device
        source, parser_source, weights = Path(source), Path(parser_source), Path(weights)
        sys.path.insert(0, str(source))
        from networks.UnetG import UnetG
        from networks.arcface_models import resnet101
        # Only constructor fields consumed by the author's generator.
        self.generator = UnetG(dict(input_nc=3, output_nc=3, ngf=32, G_norm_type='in', style_dim=64))
        checkpoint = torch.load(weights/'model.pth', map_location='cpu', weights_only=True)
        self.generator.load_state_dict(checkpoint['netG'], strict=True)
        del checkpoint
        self.arcface = resnet101()
        self.arcface.load_state_dict(torch.load(weights/'BEST_checkpoint_r101.pth', map_location='cpu', weights_only=True), strict=True)
        sys.path.insert(0, str(parser_source))
        from model import BiSeNet
        from resnet import Resnet18
        # The complete parser checkpoint immediately replaces every learned weight.
        # Skip the author's redundant ImageNet download during construction only.
        initialize = Resnet18.init_weight
        try:
            Resnet18.init_weight = lambda _: None
            self.parser = BiSeNet(n_classes=19)
        finally:
            Resnet18.init_weight = initialize
        self.parser.load_state_dict(torch.load(weights/'79999_iter.pth', map_location='cpu', weights_only=True), strict=True)
        for module in [self.generator, self.arcface, self.parser]:
            module.to(device).eval()
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    @torch.no_grad()
    def reference_labels(self, reference):
        # Author parser preprocessing: RGB 512x512, ImageNet mean/std.
        normalized = (rgb_tensor(reference, 512, self.device) + 1) / 2
        logits = self.parser((normalized-self.mean)/self.std)[0]
        labels = logits.argmax(1, keepdim=True).float()
        return F.interpolate(labels, size=(256, 256), mode='nearest').long()

    @torch.no_grad()
    def native(self, observed, mask, reference, labels):
        if observed.shape != (1, 3, 256, 256) or reference.shape != observed.shape:
            raise ValueError('Author generator requires batch-one 256px RGB tensors')
        if mask.shape != (1, 1, 256, 256) or labels.shape != mask.shape:
            raise ValueError('Native mask/reference-label geometry mismatch')
        if not ((mask == 0) | (mask == 1)).all() or labels.min() < 0 or labels.max() > 18:
            raise ValueError('Invalid missing mask or reference labels')
        if not torch.isfinite(observed).all() or not torch.isfinite(reference).all():
            raise ValueError('Nonfinite input RGB')
        reference_id = self.arcface(F.interpolate(reference, size=112, mode='bilinear', align_corners=False))
        segmentation = F.one_hot(labels[:, 0], num_classes=19).permute(0, 3, 1, 2).float()
        masked = observed * (1-mask)
        raw, _, _, _, _ = self.generator(torch.cat((masked, mask), dim=1), reference_id,
                                        reference, segmentation, mask, train_mode='inpainting')
        if not torch.isfinite(raw).all():
            raise ValueError('Author generator returned a nonfinite image')
        return raw

    @torch.no_grad()
    def reconstruct(self, observed, missing, reference):
        original = np.asarray(observed.convert('RGB')).copy()
        if original.shape[:2] != missing.shape or original.shape[0] != original.shape[1]:
            raise ValueError('This research screen expects matched square observed RGB/mask')
        safe_observation = np.where(missing[..., None], np.uint8(128), original)
        image = rgb_tensor(Image.fromarray(safe_observation), 256, self.device)
        ref = rgb_tensor(reference, 256, self.device)
        mask = conservative_mask(missing).to(self.device)
        labels = self.reference_labels(reference)
        raw = self.native(image, mask, ref, labels)
        # Upsampling does not add native detail; exact visible bytes are restored last.
        array = ((raw[0].clamp(-1, 1)+1)*127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        native = Image.fromarray(array)
        resized = np.asarray(native.resize(observed.size, Image.Resampling.BILINEAR))
        result = np.where(missing[..., None], resized, original)
        meta = {'native_size': 256, 'output_size': list(observed.size),
                'native_mask_pixels': int(mask.sum()), 'original_mask_pixels': int(missing.sum()),
                'reference_label_classes': sorted(int(x) for x in labels.unique().cpu().tolist()),
                'known_pixels_unchanged': bool(np.array_equal(result[~missing], original[~missing]))}
        return Image.fromarray(result), native, meta
