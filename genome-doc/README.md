# Genome-Doc

## Hallucination-Free Document Restoration via Symbolic Genome Inference and Conditional Neural Re-Rendering

**Conference Target:** CVPR / ICCV / ECCV

---

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Architecture

Genome-Doc comprises **three modules** in a single forward pass:

| Module | Purpose | Params |
|--------|---------|--------|
| **DGI** (Document Genome Inferrer) | Extracts structured symbolic spec from degraded image | ~250M |
| **SIR** (Style & Identity Refiner) | Captures document visual identity from clean patches | ~25M |
| **NRE** (Neural Re-Rendering Engine) | Generates clean document from genome + style | ~360M |

## Project Structure

```
genome-doc/
├── configs/                    # YAML configs for all modules
├── genome/                     # Document Genome schema & rendering
│   ├── schema.py              # Pydantic v2 genome specification
│   ├── utils.py               # Genome manipulation & comparison
│   └── renderer.py            # Skeleton image renderer
├── data/                       # Synthetic data generation
├── models/                     # Neural network modules
│   ├── dgi/                   # Document Genome Inferrer
│   ├── sir/                   # Style & Identity Refiner
│   └── nre/                   # Neural Re-Rendering Engine
├── training/                   # Training scripts & losses
├── inference/                  # Inference pipelines
├── eval/                       # Metrics & benchmarking
└── requirements.txt
```

## Training With Existing Cloud Data

```bash
# Stage 1: Mount or copy your existing backed-up dataset.
# Expected structure:
#   DATA_ROOT/train/images/clean
#   DATA_ROOT/train/images/degraded
#   DATA_ROOT/train/genomes
#   DATA_ROOT/val/images/clean
#   DATA_ROOT/val/images/degraded
#   DATA_ROOT/val/genomes
export DATA_ROOT=/content/drive/MyDrive/prometheus/data/synthetic

# Stage 2: Train SIR (style encoder)
python training/train_sir.py \
  --config configs/sir_resnet.yaml \
  --data-dir $DATA_ROOT/train \
  --val-dir $DATA_ROOT/val

# Stage 3: Train DGI (genome extractor)
python training/train_dgi.py \
  --config configs/dgi_donut.yaml \
  --data-dir $DATA_ROOT/train \
  --val-dir $DATA_ROOT/val

# Stage 4: Train NRE (neural renderer)
python training/train_nre.py --config configs/nre_controlnet.yaml \
  --data-dir $DATA_ROOT/train \
  --val-dir $DATA_ROOT/val \
  --sir-checkpoint checkpoints/sir/best_model.pt
```

DGI now requires `images/degraded` by default. Use `--allow-clean-fallback`
only for a smoke test, not for real training. NRE now requires a real SIR
checkpoint by default; `--allow-random-style` is also smoke-test only.

## Inference

```bash
# Restore a single document
python inference/restore.py --input degraded.png --output restored.png

# Export genome as JSON
python inference/export_genome.py --input degraded.png --output genome.json
```

## Hardware Requirements

| GPU | Training Time | VRAM |
|-----|--------------|------|
| T4 (16GB) | ~7-10 days | Tight for NRE |
| L4 (24GB) | ~4-6 days | Comfortable |
| RTX 4090 (24GB) | ~3-4 days | Comfortable |
| A100 (40GB) | ~1-2 days | Plenty |

## License

MIT
