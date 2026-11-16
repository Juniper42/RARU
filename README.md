# RARU: Efficient and Utility-Preserving Recommendation Unlearning via Localized Rank-Aware Salience

Official implementation of the paper *Efficient and Utility-Preserving
Recommendation Unlearning via Localized Rank-Aware Salience*.

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

## Method at a Glance

RARU has two phases (Algorithm 1 of the paper).

1. **Salience-guided scope localization (Section 4.1).** For every entity `v`
   in the target entity set `V_t`, the salience combines collaborative
   connectivity `S_conn(v) = log(1 + deg_v) / log(1 + deg_max)` (Eq. 5) and
   representation similarity `S_sim(v) = max_e [cos(v, e)]_+` over the target
   entities of the same type (Eq. 6). The two terms are averaged into `S(v)`
   (Eq. 7). Starting from the retained neighbors of `V_t`, a max-priority queue
   expands `V_S` and stops when the top salience falls below
   `tau = S_max / k` (Eq. 8). The scope of interactions is the closed set
   `D_S = D_t union {(u,i) in D_r | u in V_S or i in V_S}` (Eq. 9).
2. **Salience-weighted ranking unlearning (Section 4.2).** With
   `S_ui = (S(u) + S(i)) / 2` (Eq. 7) and the target gradient
   `g_t = grad_{Theta_S} L(D_t; Theta)` (Eq. 10), RARU solves

   ```
   (Htilde_{r,S} + P_S + lambda I) Delta = g_t
   ```

   where `P_S = sum S_ui grad m grad m^T` (Eq. 13) stabilizes the retained
   margins `m_uij = yhat_ui - yhat_uj` (Eq. 2) and `Htilde_{r,S}` (Eq. 14) is a
   Gauss-Newton curvature built from the retained interactions in the scope.
   The system is solved with `T` conjugate-gradient iterations that never form
   the matrix, and only the embedding blocks of the entities in `V_S` are
   updated: `Theta'_S = Theta_S + Delta`, `Theta'_{\S} = Theta_{\S}` (Eq. 17).

The default configuration reproduces the released update that produced the
numbers reported in the paper. Pass `--paper_exact` for the paper-faithful
variant, which additionally

* uses both endpoints of the target interactions as `V_t` (not only the items),
* removes the `--rarl_max_scope_size` cap on the scope,
* builds `C_u` as the `2k` highest-scoring unobserved items of `u` (Section 4.2),
  instead of drawing a single random competitor per user,
* scores the curvature with the same negative samples as `g_t`,
* uses the Gauss-Newton form of `P_S` and `Htilde_{r,S}` (Eqs. 13-14), with
  `w_p = sigma(m) sigma(-m)` for BPR backbones and `w_p = sigma(1 - sigma)` for
  the BCE-trained NeuMF and WMF backbones,
* solves the linear system on `Theta_S` only,
* samples the unlearning request as a fraction of the *training interactions*
  (interaction unlearning) or as the interactions of randomly drawn items whose
  total reaches that budget (entity unlearning),
* propagates over the retained graph after unlearning, as described in
  Section 4.2.

`--paper_exact` does not rewrite the training hyperparameters. Pass the settings
of Section 5.1.6 explicitly when reproducing the paper:
`--emb_dim 64 --batch_size 1024 --test_size 0.1 --val_size 0.1`.

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

## Datasets

Place the raw files as follows:

```
data/
├── ml-1m/ratings.dat                     # MovieLens-1M
├── ml-10m/ratings.dat                    # MovieLens-10M
├── netflix/combined_data_1.txt ...       # Netflix Prize
├── yelp2018/total.txt                    # space-separated user item item ...
└── amazon-book/total.txt
```

MovieLens-1M and MovieLens-10M use the `user::item::rating::timestamp` format.
The Netflix files use the original format, where a `MovieID:` header line is
followed by `UserID,Rating,Date` rows. By default a rating is a positive
interaction when it is at least 3, and users and items with fewer than 10
interactions are removed iteratively; both thresholds are controlled by
`--min_rating` and `--min_interactions`. The statistics reported in Table 1 of
the paper are the raw dataset statistics.

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

Run the baselines by replacing `--method` with one of `retrain`, `sisa`,
`certified_removal`, `recraser`, `ultrare`, `gsgcf-ru`, `ifru`, `rrl`,
`unlearnrec` or `gferaser`, and use `--num_runs 10` to average over the ten
seeds reported in the paper. Every run prints NDCG@K, Recall@K, precision@K,
the unlearning time, URR, RDR and the MI-AUC attack metrics, and appends a row
to the CSV given by `--output_result_path`.

## Notes on Reproducibility

* `RARU.py` keeps the released update as the default and adds the
  paper-faithful variant behind `--paper_exact`. The two paths share the same
  scope construction code, so the ablation switches apply to both.
* Hyperparameters are never rewritten implicitly: the defaults in
  `parameters.py` are the released ones, so a run that reproduces Section 5.1.6
  has to pass `--emb_dim 64 --batch_size 1024 --test_size 0.1 --val_size 0.1`.
* The baseline implementations follow the released code of the respective
  papers; where a baseline reproduces only part of the original pipeline, the
  docstring in `unlearning_func/__init__.py` says which stage is used.
* Graph backbones propagate over the retained training graph. In the default
  mode the evaluation graph is the one produced by `main.compute_edges_to_remove`;
  under `--paper_exact` the target edges are deleted from the graph before the
  post-unlearning evaluation and the rank-change metrics.

## Citation

```bibtex
@inproceedings{raru2027,
  title     = {Efficient and Utility-Preserving Recommendation Unlearning via
               Localized Rank-Aware Salience},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM International Conference on Web Search and
               Data Mining (WSDM)},
  year      = {2027}
}
```
