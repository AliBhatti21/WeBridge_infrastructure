# model.py
import io
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

# ── Configuration ─────────────────────────────────────────────
CHECKPOINTS = [
    ('K2_effnetv2s_384px.pt',      'tf_efficientnetv2_s', 384, 1.0),
    ('K2_dividemix__no_ssl.pt',    'tf_efficientnetv2_s', 384, 1.0),
    ('K2_dividemix_ssl.pt',        'tf_efficientnetv2_s', 384, 2.0),
    ('K2_effnet_b0_imagenet.pt',   'efficientnet_b0',     224, 1.0),
]

BALANCED_THR = {0: 0.60, 1: 0.45, 2: 0.43}  # never None in deployment
CLASS_NAMES  = ['Low', 'Medium', 'High']
K2_VALUES    = {0: 0.2, 1: 0.5, 2: 1.0}

# ── Model architecture ─────────────────────────────────────────
def build_model(arch: str) -> nn.Sequential:
    backbone = timm.create_model(
        arch, pretrained=False, num_classes=0, global_pool='avg'
    )
    feat_dim = backbone.num_features
    return nn.Sequential(
        backbone,
        nn.BatchNorm1d(feat_dim),
        nn.Dropout(0.4),
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.2),
        nn.Linear(128, 3)
    )

def load_weights(model: nn.Sequential, ckpt_path: str) -> nn.Sequential:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif 'net_A_state' in ckpt:
        model.load_state_dict(ckpt['net_A_state'])
    elif 'net_states' in ckpt:
        model.load_state_dict(ckpt['net_states'][0]['model_state'])
    else:
        model.load_state_dict(ckpt)
    return model

# ── Called once at startup ─────────────────────────────────────
def load_all_models() -> list:
    """
    Load all checkpoints into memory.
    Returns a list of (model, img_size, weight, name).
    Call this once at server startup — not on every request.
    """
    loaded = []
    for ckpt_path, arch, img_size, weight in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f" Skipped (not found): {ckpt_path}")
            continue
        model = build_model(arch)
        model = load_weights(model, ckpt_path)
        model.eval()  
        loaded.append((model, img_size, weight, os.path.basename(ckpt_path)))
        print(f" Loaded: {ckpt_path}")

    if not loaded:
        raise RuntimeError("No checkpoints found. Check CHECKPOINTS paths.")

    print(f"\n  Ensemble ready: {len(loaded)} models loaded on CPU.")
    return loaded

# ── Called on every request ────────────────────────────────────
def predict(image_bytes: bytes, loaded_models: list) -> dict:
    """
    Run ensemble inference on a single image (as raw bytes).
    Returns a dict with probabilities and predictions.
    """
    total_probs = np.zeros(3)
    total_w     = 0.0

    for model, img_size, weight, name in loaded_models:
        # Build transform for this model's expected input size
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Decode bytes -> PIL image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Forward pass — no gradients needed for inference
        tensor = transform(img).unsqueeze(0)  # shape: (1, C, H, W)
        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().numpy()

        total_probs += probs * weight
        total_w     += weight

    # Weighted average across ensemble
    ensemble_probs = total_probs / total_w

    # Raw prediction — plain argmax
    pred_raw = int(ensemble_probs.argmax())

    # Balanced prediction — threshold-adjusted argmax
    adjusted = np.array([ensemble_probs[c] - BALANCED_THR[c] for c in range(3)])
    pred_bal = int(adjusted.argmax())

    return {
        'prediction_balanced': CLASS_NAMES[pred_bal],
        'prediction_raw':      CLASS_NAMES[pred_raw],
        'k2_value':            K2_VALUES[pred_bal],
        'probabilities': {
            'Low':    round(float(ensemble_probs[0]), 4),
            'Medium': round(float(ensemble_probs[1]), 4),
            'High':   round(float(ensemble_probs[2]), 4),
        },
        'thresholds_applied': BALANCED_THR,
        'models_used': len(loaded_models),
    }