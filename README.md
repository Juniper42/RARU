# RARU: Efficient and Utility-Preserving Recommendation Unlearning via Localized Rank-Aware Salience

## Abstract

Historical interactions often have to be removed from a recommender system on
demand, for privacy requests, data correction, or item discontinuation, which
has motivated the study of recommendation unlearning. Existing recommendation
unlearning methods largely inherit the objective of general machine unlearning
by approximating the effect of retraining on the retained data. This
retraining-aligned formulation does not match the ranking-centric nature of
recommendation, where utility is reflected more directly by the top-ranked
results than by parameter-level proximity to a retrained reference. We argue
that efficient recommendation unlearning should reduce the ranking influence of
the target interactions while preserving the utility of the retained ones.
Building on this insight, we propose **Localized Rank-Aware Salience
Recommendation Unlearning** (RARU), a model-agnostic framework that performs
ranking-targeted approximate unlearning while preserving recommendation
utility. RARU estimates interaction salience from collaborative connectivity and
representation similarity, expands a compact local scope around the target
interactions along the salience ordering, and applies a curvature-aware update
that counteracts the training contribution of the target interactions while
stabilizing the high-salience retained ranking relations. Experiments on five
datasets and three recommendation models show that RARU matches the
recommendation quality of full retraining and is, on average, 2.7x faster than
the strongest efficient baseline.

## Code Structure

```
RARU_code/
├── data/                       # datasets go here (see below)
├── models/                     # LightGCN, GAT, GraphSAGE, GMF, MLP, NeuMF, WMF, ...
├── unlearning_func/
│   ├── RARU.py                 # proposed method (Algorithm 1)
│   ├── Retrain.py              # gold-standard retraining
│   ├── SISA.py
│   ├── CertifiedRemoval.py
│   ├── RecEraser.py
│   ├── UltraRE.py
│   ├── GSGCF_RU.py             # LightGCN only
│   ├── IFRU.py
│   ├── RRL.py                  # BPR-trained backbones only
│   ├── UnlearnRec.py
│   └── GFEraser.py
├── attack.py                   # membership-inference attack (MI-AUC)
├── data_loader.py              # dataset download parsing, k-core filter, splits
├── evaluate.py                 # NDCG@K / Recall@K / Precision@K
├── unlearning_metrics.py       # URR and RDR (rank change of the targets)
├── logger.py
├── main.py                     # experiment driver
├── model_cache.py
├── parameters.py               # command-line arguments
├── trainer.py
├── utils.py
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

Run RARU with LightGCN on ML-1M with the released update:

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

Run the paper-faithful variant (paper hyper-parameters and request construction):

```bash
python main.py \
    --method raru \
    --dataset ml-1m \
    --backbone lightgcn \
    --unlearning_task interaction \
    --unlearning_ratio 0.1 \
    --scope_cutoff_k 10 \
    --damping_lambda 0.01 \
    --cg_steps 30 \
    --emb_dim 64 \
    --batch_size 1024 \
    --test_size 0.1 \
    --val_size 0.1 \
    --paper_exact
```