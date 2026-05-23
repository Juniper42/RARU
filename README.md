# RARU: Efficient and Utility-Preserving Recommendation Unlearning via Localized Rank-Aware Salience

## Abstract

Existing recommendation unlearning methods largely inherit the objective of general machine unlearning by approximating the effect of retraining on retained data. However, this retraining-aligned formulation overlooks the ranking-centric nature of recommendation tasks, where utility is determined by top-ranked results rather than parameter-level proximity to a retrained reference. Therefore, efficient recommendation unlearning should reduce the ranking influence of target interactions while preserving the utility of retained interactions.
Building on this insight, we propose **Rank-Aware Recommendation Unlearning** (RARU), a model-agnostic framework for efficient and utility-preserving ranking-targeted approximate unlearning. Specifically, RARU first computes interaction salience from collaborative connectivity and representation similarity to identify locally rank-relevant interactions. Guided by this salience, it expands a compact localized scope and performs a curvature-aware update that counteracts the training contribution of target interactions while stabilizing high-salience retained ranking relations. Across five datasets and three recommendation models, RARU is about 2.5$\times$ faster on average than the strongest efficient baseline, while matching the recommendation quality of full retraining. 

## Code Structure

```
RARU_code/
├── data/
├── models/
├── unlearning_func/
│   ├── RARU.py
│   ├── Retrain.py
│   ├── SISA.py
│   ├── CertifiedRemoval.py
│   ├── RecEraser.py
│   ├── UltraRE.py
│   ├── GSGCF_RU.py
│   ├── IFRU.py
│   ├── RRL.py
│   ├── UnlearnRec.py
│   └── GFEraser.py
├── attack.py
├── data_loader.py
├── evaluate.py
├── unlearning_metrics.py
├── main.py
├── parameters.py
├── trainer.py
├── requirements.txt
└── README.md
```

## Setup and Installation

1. **Create a conda environment**:
    ```bash
    conda create -n raru python=3.10
    conda activate raru
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

Run the proposed method (RARU) with LightGCN on ML-1M:

```bash
python main.py \
    --method raru \
    --dataset ml-1m \
    --backbone lightgcn \
    --unlearning_task interaction \
    --unlearning_ratio 0.1 \
    --scope_cutoff_k 10 \
    --damping_lambda 0.01 \
    --cg_steps 30
```
