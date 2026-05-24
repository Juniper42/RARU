# RARU: Efficient and Utility-Preserving Recommendation Unlearning via Localized Rank-Aware Salience

## Abstract

In recommender systems, historical interactions often need to be removed on demand for privacy requests, data correction, or item discontinuation, which has motivated the study of recommendation unlearning. Existing recommendation unlearning methods largely inherit the objective of general machine unlearning by approximating the effect of retraining on retained data. However, this retraining-aligned formulation is not fully aligned with the ranking-centric nature of recommendation, where utility is reflected more directly by top-ranked results than by parameter-level proximity to a retrained reference. We argue that efficient recommendation unlearning should reduce the ranking influence of target interactions while preserving the utility of retained interactions. Building on this insight, we propose **Rank-Aware Recommendation Unlearning** (RARU), a model-agnostic framework that performs ranking-targeted approximate unlearning while preserving recommendation utility. Specifically, RARU first estimates interaction salience from collaborative connectivity and representation similarity. Guided by this salience, it expands a compact local scope around the target and applies a curvature-aware update that counteracts the training contribution of target interactions while stabilizing high-salience retained ranking relations. Experiments on five datasets and three recommendation models show that RARU achieves recommendation quality on par with full retraining and is, on average, about 2.5$\times$ faster than the strongest efficient baseline. 

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
