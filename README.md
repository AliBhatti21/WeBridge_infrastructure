# WeBridge Infrastructure Project

Deep learning pipeline for automated bridge inspection image classification, developed for **Sarix Srl**. The system classifies structural defect severity from inspection imagery and flags non-conforming images, using ensembles of deep learning models trained on an HPC cluster.

---

## Overview

The pipeline addresses two related tasks on bridge inspection imagery:

1. **K2 Severity Classification** — classifies structural defect severity into three ordinal classes:
   - **Low** (0.2)
   - **Medium** (0.5)
   - **High** (1.0)
2. **Non-Conformity (NC) Detection** — a binary classifier that flags non-conforming images (e.g., irrelevant, low quality, or invalid captures), applied specifically to images classified as High severity.

---

## Model Architecture

### K2 Severity Classifier
- **Backbone(s):** EfficientNetV2-S and EfficientNet-B0, combined in an ensemble
- **Training data:** ~12,000 labeled inspection images
- **Ensemble strategy:** 4-network ensemble with seed offsets 
- **Threshold optimization:** balanced ordinal penalty matrix used to tune decision thresholds across the Low/Medium/High severity boundaries

### Non-Conformity (NC) Classifier
- **Backbone:** EfficientNetB0
- **Input resolution:** 224px
- **Scope:** applied only to images predicted as High severity by the K2 classifier

### Self-Supervised Pretraining
- **Method:** SimCLR (Simple Contrastive Learning Representations)
- **Data:** ~44,000 unlabeled inspection images
- Used to initialize backbone weights prior to supervised fine-tuning

### Semi-Supervised / Noisy-Label Experiments
- **Method:** DivideMix co-training
- **Status:** Currently configured as **warmup-only** — full DivideMix training was found to collapse under class imbalance and was disabled pending further tuning

---

## Repository Structure

```
webridge-inspection/
├── checkpoints/                # (gitignored) model weights
├── src/
│   ├── training/
│   │   ├── simclr_pretrain.py
│   │   ├── k2_train.py
│   │   ├── nc_train.py
│   │   └── dividemix_warmup.py
│   ├── ensemble/
│   │   └── K2_4net_ensemble.py
│   ├── evaluation/
│   │   └── threshold_optimization.py
│   └── inference/
│       └── fastapi_server.py    # ensemble inference API
├── slurm/                       # SLURM job scripts for HPC training
├── configs/
├── requirements.txt
└── README.md
```

---

## Training Infrastructure

- **Cluster:** HPC cluster via SLURM
- **GPU:** NVIDIA H200
- **Environment:** `WeBridge_venv` (Python virtual environment)

### Running a training job
```bash
sbatch slurm/train_k2.slurm
```

Make sure `WeBridge_venv` is activated within the job script and that dataset paths point to the HPC-mounted data directories, not local paths.

---

## Inference

A **FastAPI** server exposes the K2 ensemble for inference:

- All four ensemble checkpoints are loaded into memory at startup for low-latency prediction
- Input: inspection image
- Output: severity class (Low/Medium/High) with associated confidence, plus NC flag when severity is High

```bash
uvicorn src.inference.fastapi_server:app --host 0.0.0.0 --port 8000
```

Example request:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@sample_image.jpg"
```

*(Update the endpoint/schema above to match your actual FastAPI route definitions.)*

---

## Setup

```bash
git clone https://github.com/<org-or-user>/webridge-inspection.git
cd webridge-inspection

python3 -m venv WeBridge_venv
source WeBridge_venv/bin/activate
pip install -r requirements.txt
```

---

## Known Issues / In Progress


- Threshold optimization matrix is tuned for the current label distribution — revisit if class balance shifts
- AWS migration in progress: moving training to **SageMaker Training Jobs** (parallel GPU training of the 4 ensemble models) and inference to a **Lambda + API Gateway** CPU-based architecture

---

## Team / Contacts

- **Project lead:** Ali
- **Company:** Sarix Srl


---

## License

Proprietary — client work for Sarix Srl. Not for public distribution without written permission.
