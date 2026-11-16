"""Unlearning methods compared in the paper.

Each wrapper takes the model trained on the full data and the prepared ``data``
object and returns ``(unlearned_model, removed_edges, unlearning_time)``.  RARU
is the proposed method; the other wrappers implement the baselines."""

import time

from logger import Logger

# Import implemented unlearning methods.
from .CertifiedRemoval import CertifiedRemoval
from .GFEraser import GFEraser
from .GSGCF_RU import GSGCF_RU
from .IFRU import IFRU
from .RARU import RARU
from .RecEraser import RecEraser
from .Retrain import Retrain
from .RRL import RRL
from .SISA import SISA
from .UltraRE import UltraRE
from .UnlearnRec import UnlearnRec

logger = Logger.get_logger("Unlearning")


def retrain_unlearning(original_model, data, args, device, user_train_items=None):
    """Retrain from scratch on the retained interactions D_r (gold standard)."""
    logger.info("Starting unlearning via retraining...")
    return Retrain(original_model, data, args, device).unlearn(user_train_items)


def gsgcf_ru_unlearning(original_model, data, args, device):
    """GSGCF-RU: graph-based baseline reported for LightGCN only in the paper."""
    logger.info("Starting unlearning via GSGCF-RU...")
    return GSGCF_RU(original_model, data, args, device).unlearn()


def receraser_unlearning(original_model, data, args, device):
    """RecEraser: partitions the interactions into shards, retrains the affected
    shards and aggregates their predictions with learned attention weights.
    """
    logger.info("Starting unlearning via RecEraser...")
    return RecEraser(original_model, data, args, device).unlearn()


def raru_unlearning(original_model, data, args, device):
    """RARU: SGSL scope (Section 4.1) + SWRU update (Section 4.2)."""
    logger.info("Starting unlearning via RARU...")
    return RARU(original_model, data, args, device).unlearn()


def ifru_unlearning(original_model, data, args, device):
    """IFRU: inverse-Hessian-vector-product update on the retained interactions."""
    logger.info("Starting unlearning via IFRU...")
    return IFRU(original_model, data, args, device).unlearn()


def sisa_unlearning(original_model, data, args, device):
    """SISA: retrains only the shards that contain a target interaction."""
    logger.info("Starting unlearning via SISA...")
    return SISA(original_model, data, args, device).unlearn()


def certified_removal_unlearning(original_model, data, args, device):
    """CertifiedRemoval: one-step Newton removal under its original assumptions."""
    logger.info("Starting unlearning via Certified Removal...")
    return CertifiedRemoval(original_model, data, args, device).unlearn()


def ultrare_unlearning(original_model, data, args, device):
    """UltraRE: group-wise baseline that re-trains the affected groups and aggregates them."""
    logger.info("Starting unlearning via UltraRE...")
    return UltraRE(original_model, data, args, device).unlearn()


def rrl_unlearning(original_model, data, args, device):
    """RRL: reverse BPR on the targets, regularized by a diagonal Fisher estimate."""
    logger.info("Starting unlearning via RRL...")
    return RRL(original_model, data, args, device).unlearn()


def unlearnrec_unlearning(original_model, data, args, device):
    """UnlearnRec: fine-tuning with a forgetting loss, an alignment loss and BPR on D_r."""
    logger.info("Starting unlearning via UnlearnRec...")
    return UnlearnRec(original_model, data, args, device).unlearn()


def gferaser_unlearning(original_model, data, args, device):
    """GFEraser: learned embedding filter (negative-graph BPR, JS, InfoNCE, attentive BPR)."""
    logger.info("Starting unlearning via GFEraser...")
    return GFEraser(original_model, data, args, device).unlearn()
