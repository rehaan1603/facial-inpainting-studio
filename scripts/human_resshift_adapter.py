"""Explicit opt-in loader for the separate local Human Faces GPU pilot."""
import json
import torch
from resshift_adapter import ResShiftFace
from train_human_resshift_v1 import OUT, ROOT, attach, load_state, sha


class HumanResShiftFace(ResShiftFace):
    def __init__(self):
        receipt = OUT/'training.json'
        weights = OUT/'adapter_selected.pt'
        if not receipt.is_file() or not weights.is_file():
            raise ValueError('The local Human Faces adapter is unavailable. Choose the original ResShift model.')
        training = json.loads(receipt.read_text())
        protocol = json.loads((OUT/'protocol.json').read_text())
        if training['status'] != 'complete' or sha(weights) != training['adapter_sha256']:
            raise ValueError('The local trained adapter failed verification.')
        if sha(OUT/'protocol.json') != training['protocol_sha256']:
            raise ValueError('The adapter training protocol changed.')
        super().__init__()
        base = json.loads((ROOT/'research/resshift_provenance.json').read_text())
        if base['weights_sha256'] != training['base_provenance']['weights_sha256']:
            raise ValueError('The adapter requires its original pretrained model weights.')
        attach(self.model, protocol['rank'])
        load_state(self.model, torch.load(weights, map_location='cpu', weights_only=True))
        self.model.requires_grad_(False)
        self.training_metadata = dict(adapter_sha256=training['adapter_sha256'],
            training_protocol_sha256=training['protocol_sha256'],steps=training['selected_step'],
            training_photographs=training['unique_training_photographs'],
            status='Experimental local adaptation; exact hidden identity is not verified.')
