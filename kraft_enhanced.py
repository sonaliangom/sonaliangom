#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRAFT: Enhanced RuleTransNet with leakage-free CV, device fixes, FDR stats, improved rules, calibration, and ablations
Author: Russell Longjam (Enhanced, revised)
Date: August 2025
"""

import os, re, sys, time, math, warnings, random
from contextlib import nullcontext

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from scipy.stats import wilcoxon
from scipy.stats import t as student_t
import shap
import psutil

# ─────────────────────── Global config and device handling ─────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Favor determinism for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# Device: prefer MPS on Mac, else CUDA:0, else CPU. Keep as torch.device throughout.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

if DEVICE.type == "cuda":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

USE_AMP = True  # Automatic Mixed Precision (CUDA and MPS via torch.autocast)
PIN_MEMORY = DEVICE.type == "cuda"
NUM_WORKERS = 0  # keep deterministic

N_SPLITS = 5
EPOCHS_OPTUNA = 200
EPOCHS_FINAL = 600
PATIENCE = 20
BATCH_SIZE = 256
N_TRIALS_OPTUNA = 200
GRADIENT_CLIP_NORM = 1.0
MIN_LEARNING_RATE = 1e-4
MAX_EMBEDDING_DIM = 256

# Logging tee
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, open("KRAFT1.txt", "w"))

# ───────────────────── Human-readable rule utilities ──────────────────────────

def describe_rule(rule, feature_names):
    parts = []
    for (idx, op, th) in rule:
        name = feature_names[idx] if idx < len(feature_names) else f"feat[{idx}]"
        op_str = "≤" if op == "<=" else ">"
        parts.append(f"{name} {op_str} {th:.4f}")
    return " AND ".join(parts)


def print_human_rules(rules, feature_names):
    print("\n=== HUMAN-READABLE DECISION RULES ===")
    for i, r in enumerate(rules):
        print(f"Rule {i+1}: {describe_rule(r, feature_names)}")
    print("=" * 50)


def plot_rule_coverage(X, y, rules, feature_names, dataset, fold):
    if not rules:
        return
    acts = apply_enhanced_rules(X, rules, feature_names)
    n_rules = min(len(rules), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(n_rules):
        try:
            sns.violinplot(x=y, y=acts[:, i], ax=axes[i])
            axes[i].set_title(f'Rule {i+1} Coverage')
            axes[i].set_xlabel("Class Label")
            axes[i].set_ylabel(f'Rule {i+1} Activation')
        except Exception as e:
            print(f"Error plotting rule {i+1}: {e}")
            axes[i].text(0.5, 0.5, f'Error plotting\nRule {i+1}', ha='center', va='center', transform=axes[i].transAxes)
    for i in range(n_rules, 6):
        axes[i].set_visible(False)
    plt.suptitle(f'Rule Coverage Analysis - {dataset} Fold {fold}')
    plt.tight_layout()
    plt.show()
    plt.close()

# ───────────────────────── Model components ───────────────────────────────────

class StableFeatureTokenizer(nn.Module):
    def __init__(self, n_cont, cat_cards, d_emb):
        super().__init__()
        self.n_cont = n_cont
        self.d_emb = d_emb
        if n_cont:
            self.cont_emb = nn.Sequential(
                nn.Linear(1, d_emb), nn.LayerNorm(d_emb), nn.Dropout(0.1)
            )
        else:
            self.cont_emb = None
        if cat_cards:
            self.cat_embs = nn.ModuleList([
                nn.Embedding(min(card, 100), d_emb) for card in cat_cards
            ])
            self.cat_norm = nn.LayerNorm(d_emb)
        else:
            self.cat_embs = None
            self.cat_norm = None

    def forward(self, x_cont, x_cat):
        toks = []
        if self.n_cont and x_cont.numel() > 0:
            x_cont_clamped = torch.clamp(x_cont, -10, 10)
            cont_tok = self.cont_emb(x_cont_clamped.unsqueeze(-1))
            toks.append(cont_tok)
        if self.cat_embs and x_cat.numel() > 0:
            x_cat_clamped = torch.clamp(x_cat, 0, 99)
            cat_tok = torch.stack([emb(x_cat_clamped[:, i]) for i, emb in enumerate(self.cat_embs)], dim=1)
            cat_tok = self.cat_norm(cat_tok)
            toks.append(cat_tok)
        if not toks:
            batch_size = max(x_cont.shape[0] if x_cont.numel() > 0 else 1,
                             x_cat.shape[0] if x_cat.numel() > 0 else 1)
            return torch.zeros(batch_size, 1, self.d_emb, device=x_cont.device if x_cont.numel() > 0 else x_cat.device)
        return torch.cat(toks, dim=1)


class AdaptiveRuleHead(nn.Module):
    def __init__(self, d_emb, n_rules):
        super().__init__()
        self.n_rules = n_rules
        if n_rules > 0:
            self.rule_encoder = nn.Sequential(
                nn.Linear(n_rules, d_emb), nn.LayerNorm(d_emb), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(d_emb, d_emb), nn.LayerNorm(d_emb)
            )
            self.data_encoder = nn.Sequential(
                nn.Linear(d_emb, d_emb), nn.LayerNorm(d_emb), nn.GELU(), nn.Dropout(0.1)
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(d_emb * 2, d_emb), nn.LayerNorm(d_emb), nn.Sigmoid()
            )
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.rule_importance = nn.Parameter(torch.ones(n_rules) / max(n_rules, 1))
            self.predictor = nn.Sequential(
                nn.Linear(d_emb, d_emb // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_emb // 2, 1)
            )

    def forward(self, x_rules, deep_feat):
        if self.n_rules == 0 or x_rules.numel() == 0:
            return torch.zeros(deep_feat.size(0), 1, device=deep_feat.device)
        rule_weights = F.softmax(self.rule_importance / (self.temperature.abs() + 1e-8), dim=0)
        weighted_rules = x_rules * rule_weights.unsqueeze(0)
        rule_repr = self.rule_encoder(weighted_rules)
        data_repr = self.data_encoder(deep_feat)
        alpha_stable = torch.sigmoid(self.alpha)
        combined = alpha_stable * rule_repr + (1 - alpha_stable) * data_repr
        gate_input = torch.cat([rule_repr, data_repr], dim=-1)
        gate_weights = self.fusion_gate(gate_input)
        gated_output = combined * gate_weights
        return self.predictor(gated_output)


class MultiScaleTransformer(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        n_heads = min(n_heads, max(1, embed_dim // 4))
        self.local_processing = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
                dropout=0.1, activation='gelu', batch_first=True, norm_first=True
            ),
            num_layers=n_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, return_attn=False):
        if x.numel() == 0:
            return x if not return_attn else (x, None)
        local_out = self.local_processing(x)
        if return_attn:
            attn_outputs = []
            src = x
            for layer in self.global_transformer.layers:
                attn_out = layer.self_attn(src, src, src)[1]
                attn_outputs.append(attn_out.mean(dim=1))
                src = layer(src)
            global_out = src
            avg_attn = torch.stack(attn_outputs).mean(dim=0).detach().cpu().numpy()
            return self.output_norm(local_out + global_out), avg_attn
        else:
            global_out = self.global_transformer(x)
            return self.output_norm(local_out + global_out)


class KRAFT(nn.Module):
    def __init__(self, cat_cards, n_cont, n_rules, embed_dim, n_heads, n_layers, dropout, run_ablations=None):
        super().__init__()
        run_abl = run_ablations or {}
        embed_dim = min(embed_dim, MAX_EMBEDDING_DIM)
        self.run_abl = run_abl
        self.tokenizer = StableFeatureTokenizer(n_cont, cat_cards, embed_dim)
        if run_abl.get("no_attention"):
            self.transformer = nn.Identity()
        else:
            self.transformer = MultiScaleTransformer(embed_dim, n_heads, n_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.LayerNorm(embed_dim * 2), nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.3),
            nn.Linear(embed_dim // 2, 1)
        )
        # Rule pathways
        self.rule_head = None if (run_abl.get("no_rules") or run_abl.get("concat_rules")) else AdaptiveRuleHead(embed_dim, n_rules)
        # For concat_rules: project rule vector into a single token
        self.rules_as_token = None
        if run_abl.get("concat_rules") and n_rules > 0:
            self.rules_as_token = nn.Linear(n_rules, embed_dim)
        # Uncertainty / calibration
        if not run_abl.get("no_uncertainty"):
            self.unc_mean = nn.Linear(embed_dim, 1)
            self.unc_logvar = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Linear(embed_dim // 2, 1), nn.Tanh())
            self.unc_penalty = nn.Parameter(torch.tensor(-1.0))
            self.temperature_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.unc_mean = None
            self.unc_logvar = None
            self.unc_penalty = None
            self.temperature_scale = None

    def forward(self, x_cont, x_cat, x_rules):
        if x_cont.numel() > 0:
            x_cont = torch.tanh(x_cont / 10) * 10
            x_cont = torch.clamp(x_cont, -10, 10)
        if x_cat.numel() > 0:
            x_cat = torch.clamp(x_cat.long(), 0, 99)
        if x_rules.numel() > 0:
            x_rules = torch.clamp(x_rules, 0, 1)
        # Tokenize
        try:
            toks = self.tokenizer(x_cont, x_cat)
            if torch.isnan(toks).any() or torch.isinf(toks).any():
                toks = torch.zeros_like(toks)
        except Exception:
            batch_size = max(x_cont.shape[0] if x_cont.numel() > 0 else 1,
                             x_cat.shape[0] if x_cat.numel() > 0 else 1)
            toks = torch.zeros(batch_size, 1, self.tokenizer.d_emb, device=x_cont.device if x_cont.numel() > 0 else x_cat.device)
        # Optional concat of rules as one token
        if self.rules_as_token is not None and x_rules.numel() > 0:
            rule_tok = self.rules_as_token(x_rules).unsqueeze(1)
            toks = torch.cat([toks, rule_tok], dim=1)
        # Transformer / pooling
        if isinstance(self.transformer, MultiScaleTransformer):
            feat = self.transformer(toks).mean(dim=1)
        else:
            feat = toks.mean(dim=1)
        out = self.mlp_head(feat)
        # Temperature scaling
        if (self.temperature_scale is not None) and (not self.run_abl.get("no_temp")):
            out = out / torch.clamp(self.temperature_scale.abs(), min=0.1, max=5.0)
        # Rule head fusion
        if self.rule_head:
            try:
                rule_out = self.rule_head(x_rules, feat)
                if not (torch.isnan(rule_out).any() or torch.isinf(rule_out).any()):
                    out = out + rule_out
            except Exception:
                pass
        mean = None
        log_var = torch.tensor(0.0, device=out.device)
        if self.unc_logvar is not None:
            try:
                mean = self.unc_mean(feat)
                log_var = self.unc_logvar(feat)
                if not self.run_abl.get("no_unc_penalty"):
                    uncertainty_weight = torch.sigmoid(self.unc_penalty) * 0.1
                    out = out - uncertainty_weight * torch.clamp(log_var.exp(), 0, 10)
            except Exception:
                pass
        out = torch.clamp(out, -10, 10)
        return out, mean, log_var


class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            inputs = torch.clamp(inputs, -10, 10)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss.mean()

# ───────────────────── Training and evaluation utils ──────────────────────────

def make_pos_weight(y_np, device):
    pos, neg = float(y_np.sum()), float(len(y_np) - y_np.sum())
    pos_weight = neg / max(pos, 1e-6)
    return torch.tensor([min(pos_weight, 10.0)], dtype=torch.float32, device=device)


def safe_evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_idx, (xc, xcat, xr, tgt) in enumerate(loader):
            try:
                xc = xc.to(device) if xc.numel() > 0 else torch.empty(0, device=device)
                xcat = xcat.to(device) if xcat.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
                xr = xr.to(device) if xr.numel() > 0 else torch.empty(0, device=device)
                tgt = tgt.to(device)
                logit, _, _ = model(xc, xcat, xr)
                if torch.isnan(logit).any() or torch.isinf(logit).any():
                    continue
                loss = criterion(logit.squeeze(-1), tgt.float())
                if not (torch.isnan(loss).any() or torch.isinf(loss).any()):
                    total_loss += loss.item() * len(tgt)
                preds = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
                valid_mask = ~(np.isnan(preds) | np.isinf(preds))
                if valid_mask.any():
                    all_preds.extend(preds[valid_mask])
                    all_targets.extend(tgt.cpu().numpy()[valid_mask])
            except Exception:
                continue
    if not all_preds or not all_targets:
        return float('inf'), 0.5
    avg_loss = total_loss / max(len(all_preds), 1)
    try:
        auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5
        if np.isnan(auc) or np.isinf(auc):
            auc = 0.5
    except Exception:
        auc = 0.5
    return avg_loss, auc


def enhanced_train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, criterion,
                                 patience, epochs, device, rule_lambda=0.0, prefix=""):
    best_auc, best_epoch = 0.0, epochs
    patience_counter = 0
    best_model_state = None
    use_amp = USE_AMP and (device.type in ["cuda", "mps"])
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    warmup_epochs = min(10, epochs // 10)
    base_lr = optimizer.param_groups[0]['lr']

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        # LR warmup
        if epoch <= warmup_epochs:
            warmup_lr = base_lr * (epoch / max(warmup_epochs, 1))
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif epoch == warmup_epochs + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
        for batch_idx, (xc, xcat, xr, tgt) in enumerate(train_loader):
            try:
                xc, xcat, xr, tgt = [t.to(device) for t in (xc, xcat, xr, tgt)]
                if torch.isnan(xc).any() or torch.isnan(xr).any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()
                with autocast_ctx:
                    logit, mean, log_var = model(xc, xcat, xr)
                    if torch.isnan(logit).any() or torch.isinf(logit).any():
                        continue
                    classification_loss = criterion(logit.squeeze(-1), tgt.float())
                    kl_divergence = 0.0
                    if mean is not None:
                        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                        kl_divergence = kl_div / max(xc.size(0), 1)
                        if torch.isnan(kl_divergence) or torch.isinf(kl_divergence):
                            kl_divergence = 0.0
                    rule_loss = 0.0
                    if getattr(model, 'rule_head', None) is not None and hasattr(model.rule_head, 'alpha'):
                        try:
                            rule_activation = (xr.sum(dim=1) > 0).float()
                            model_probs = torch.sigmoid(logit.squeeze(-1))
                            consistency = torch.abs(model_probs - rule_activation)
                            rule_loss = consistency.mean()
                            alpha_reg = 0.01 * (model.rule_head.alpha - 0.5).pow(2)
                            rule_loss = rule_loss + alpha_reg
                        except Exception:
                            rule_loss = 0.0
                    total_loss = classification_loss + 0.01 * kl_divergence + rule_lambda * rule_loss
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue
                if use_amp and scaler is not None:
                    scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
                epoch_loss += float(total_loss.detach().cpu())
                valid_batches += 1
            except Exception:
                continue
        if valid_batches == 0:
            print(f"No valid batches in epoch {epoch}, stopping training")
            break
        val_loss, val_auc = safe_evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        if epoch % 25 == 0 or epoch == epochs:
            train_loss, train_auc = safe_evaluate(model, train_loader, criterion, device)
            print(f"{prefix} | Epoch {epoch:3d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Train AUC {train_auc:.4f} | Val AUC {val_auc:.4f}")
        if val_auc > best_auc and not (np.isnan(val_auc) or np.isinf(val_auc)):
            best_auc, best_epoch = val_auc, epoch
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{prefix} | Early stopping at epoch {epoch}")
                break
    if best_model_state is not None:
        # Load to current device
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
    return model, best_epoch, best_auc


def enhanced_objective_for_optuna(trial, data_dict):
    try:
        Xc_tr, Xcat_tr, R_tr, y_tr = data_dict["train"]
        Xc_va, Xcat_va, R_va, y_va = data_dict["val"]
        cat_cards = data_dict["cat_cardinalities"]
        def _clean(arr):
            if arr is None:
                return np.array([])
            if arr.dtype == object:
                arr = np.where(arr == None, np.nan, arr)
            cleaned = np.nan_to_num(arr.astype(np.float32), nan=0., posinf=1., neginf=-1.)
            return np.clip(cleaned, -10, 10)
        model_par = dict(
            embed_dim=trial.suggest_categorical("embed_dim", [64, 128, 192]),
            n_heads=trial.suggest_categorical("n_heads", [4, 8]),
            n_layers=trial.suggest_int("n_layers", 2, 4),
            dropout=trial.suggest_float("dropout", 0.05, 0.3)
        )
        opt_par = dict(
            lr=trial.suggest_float("lr", 5e-5, 2e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        )
        rule_lambda = trial.suggest_float("rule_lambda", 0.01, 0.3, log=True)
        model = KRAFT(cat_cards, Xc_tr.shape[1], R_tr.shape[1], **model_par).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), **opt_par)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=8, factor=0.5, min_lr=MIN_LEARNING_RATE)
        pos_weight = make_pos_weight(y_tr, DEVICE)
        criterion = BalancedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight, label_smoothing=0.05)
        toT = lambda arr: torch.from_numpy(_clean(arr)).float()
        batch_size_dyn = min(BATCH_SIZE, len(y_tr) // 2 + 1)
        tr_loader = DataLoader(
            TensorDataset(toT(Xc_tr), torch.from_numpy(Xcat_tr).long(), toT(R_tr), torch.from_numpy(y_tr).float()),
            batch_size=batch_size_dyn, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        va_loader = DataLoader(
            TensorDataset(toT(Xc_va), torch.from_numpy(Xcat_va).long(), toT(R_va), torch.from_numpy(y_va).float()),
            batch_size=batch_size_dyn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        _, _, best_auc = enhanced_train_and_evaluate(model, tr_loader, va_loader, optimizer, scheduler, criterion,
                                                     PATIENCE, EPOCHS_OPTUNA, DEVICE, rule_lambda, prefix="OptunaTune")
        return best_auc if not (np.isnan(best_auc) or np.isinf(best_auc)) else 0.5
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.5

# ───────────────────── Rules: extraction and application ───────────────────────

def get_enhanced_ensemble_rules(X, y, feature_names, n_estimators=40, max_depth=6, min_samples_leaf=15,
                                top_k=100, min_coverage=0.01, max_coverage=0.8, round_threshold=3, random_state=42):
    print(f"\n=== ENSEMBLE RULE EXTRACTION (RandomForest; fold-train only) ===")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        random_state=random_state, class_weight='balanced', min_impurity_decrease=0.001
    )
    rf_model.fit(X, y)
    rules_list = []
    def recurse(node, path, tree_):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_idx = int(tree_.feature[node])
            threshold = float(tree_.threshold[node])
            if 0 <= feature_idx < len(feature_names):
                tval = round(threshold, round_threshold)
                left_path = path + [(feature_idx, "<=", tval)]
                recurse(tree_.children_left[node], left_path, tree_)
                right_path = path + [(feature_idx, ">", tval)]
                recurse(tree_.children_right[node], right_path, tree_)
        else:
            if path:
                rules_list.append(tuple(path))  # preserve order
    for est in rf_model.estimators_:
        recurse(0, [], est.tree_)
    # Deduplicate exact sequences while preserving order
    seen = set()
    unique_rules = []
    for r in rules_list:
        if r not in seen:
            seen.add(r)
            unique_rules.append(list(r))
    # Coverage-based pruning
    acts = apply_enhanced_rules(X, unique_rules, feature_names)
    coverage = acts.mean(axis=0)
    keep = (coverage >= min_coverage) & (coverage <= max_coverage)
    unique_rules = [unique_rules[i] for i in range(len(unique_rules)) if keep[i]]
    if len(unique_rules) > top_k:
        print(f"Too many rules ({len(unique_rules)}), selecting top {top_k} by coverage")
        top_idx = np.argsort(coverage[keep])[-top_k:]
        unique_rules = [unique_rules[i] for i in top_idx]
    print(f"Extracted {len(unique_rules)} rules (post-prune)")
    return unique_rules


def get_single_tree_rules(X, y, feature_names, max_depth=6, min_samples_leaf=15, round_threshold=3):
    print(f"\n=== SINGLE TREE RULE EXTRACTION (fold-train only) ===")
    tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42,
                                        class_weight='balanced', min_impurity_decrease=0.001)
    tree_model.fit(X, y)
    tree_ = tree_model.tree_
    rules = []
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_idx = int(tree_.feature[node])
            threshold = round(float(tree_.threshold[node]), round_threshold)
            left_path = path + [(feature_idx, "<=", threshold)]
            recurse(tree_.children_left[node], left_path)
            right_path = path + [(feature_idx, ">", threshold)]
            recurse(tree_.children_right[node], right_path)
        else:
            if path:
                rules.append(path)
    recurse(0, [])
    print(f"Extracted {len(rules)} single-tree rules")
    return rules


def apply_enhanced_rules(X, rules, feature_names):
    if not rules or X.shape[0] == 0:
        return np.zeros((X.shape[0], max(len(rules), 1)), dtype=np.float32)
    rule_features = np.zeros((X.shape[0], len(rules)), dtype=np.float32)
    for i, rule in enumerate(rules):
        if not rule:
            rule_features[:, i] = 0.5
            continue
        mask = np.ones(X.shape[0], dtype=bool)
        for (feature_idx, operator, value) in rule:
            if 0 <= feature_idx < X.shape[1]:
                feature_data = X[:, feature_idx]
                if operator == '<=':
                    mask &= (feature_data <= value)
                else:
                    mask &= (feature_data > value)
        rule_features[:, i] = mask.astype(np.float32)
    return rule_features

# ───────────────────── Metrics, stats, and calibration ─────────────────────────

def calculate_enhanced_metrics(y_true, y_pred_bin, y_prob):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    except ValueError:
        return {k: 0.0 for k in ["Accuracy", "BalancedAcc", "AUC", "AP", "Precision", "Recall", "Specificity", "F1", "Brier"]}
    y_prob = np.ravel(y_prob)
    valid_mask = np.isfinite(y_prob)
    if not valid_mask.any():
        return {k: 0.0 for k in ["Accuracy", "BalancedAcc", "AUC", "AP", "Precision", "Recall", "Specificity", "F1", "Brier"]}
    y_true_valid = y_true[valid_mask]
    y_pred_bin_valid = y_pred_bin[valid_mask]
    y_prob_valid = y_prob[valid_mask]
    try:
        auc = roc_auc_score(y_true_valid, y_prob_valid) if len(np.unique(y_true_valid)) > 1 else 0.5
        ap = average_precision_score(y_true_valid, y_prob_valid)
        brier = brier_score_loss(y_true_valid, y_prob_valid)
    except Exception:
        auc = ap = brier = 0.5
    return dict(
        Accuracy=accuracy_score(y_true_valid, y_pred_bin_valid),
        BalancedAcc=balanced_accuracy_score(y_true_valid, y_pred_bin_valid),
        AUC=auc,
        AP=ap,
        Precision=precision_score(y_true_valid, y_pred_bin_valid, zero_division=0),
        Recall=recall_score(y_true_valid, y_pred_bin_valid, zero_division=0),
        Specificity=tn / (tn + fp) if (tn + fp) > 0 else 0,
        F1=f1_score(y_true_valid, y_pred_bin_valid, zero_division=0),
        Brier=brier,
    )


def perform_wilcoxon_test(main_metrics, other_metrics):
    if len(main_metrics) != len(other_metrics) or len(main_metrics) < 2:
        return float('nan')
    try:
        _, p_value = wilcoxon(main_metrics, other_metrics)
        return p_value
    except ValueError:
        return float('nan')


def fdr_bh(pvals):
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    prev = 0
    for i, idx in enumerate(order, start=1):
        ranked[idx] = pvals[idx] * n / i
        ranked[idx] = min(ranked[idx], 1.0)
        if ranked[idx] < prev:
            ranked[idx] = prev
        prev = ranked[idx]
    # Ensure monotonicity in reverse
    for i in range(n - 2, -1, -1):
        ranked[order[i]] = min(ranked[order[i]], ranked[order[i + 1]])
    return ranked


def calibrate_probs(y_val, prob_val, prob_test, method='auto'):
    y_val = np.asarray(y_val).astype(int)
    prob_val = np.clip(np.asarray(prob_val), 1e-6, 1 - 1e-6)
    prob_test = np.clip(np.asarray(prob_test), 1e-6, 1 - 1e-6)
    if method == 'auto':
        method = 'isotonic' if len(y_val) >= 200 else 'platt'
    if method == 'isotonic':
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(prob_val, y_val)
            return np.clip(iso.transform(prob_test), 1e-6, 1 - 1e-6)
        except Exception:
            method = 'platt'
    # Platt scaling
    logit = lambda p: np.log(p) - np.log1p(-p)
    X_val = logit(prob_val).reshape(-1, 1)
    X_test = logit(prob_test).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_val, y_val)
    return lr.predict_proba(X_test)[:, 1]

# ───────────────────── Visualization helpers (unchanged core) ─────────────────

def plot_pr_curve(y_true, y_prob, ds_name, fold, model_name):
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.title(f'Precision-Recall Curve ({model_name}) - {ds_name} Fold {fold}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(f"{ds_name}_fold{fold}_{model_name}_pr_curve.png")
        plt.show()
    except Exception as e:
        print(f"PR curve failed: {e}")


def plot_roc_curve(y_true, y_prob, ds_name, fold, model_name):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, marker='.')
        plt.title(f'ROC Curve ({model_name}) - {ds_name} Fold {fold}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(f"{ds_name}_fold{fold}_{model_name}_roc_curve.png")
        plt.show()
    except Exception as e:
        print(f"ROC curve failed: {e}")


def plot_confusion_matrix(y_true, y_pred_bin, ds_name, fold, model_name):
    try:
        cm = confusion_matrix(y_true, y_pred_bin)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({model_name}) - {ds_name} Fold {fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f"{ds_name}_fold{fold}_{model_name}_cm.png")
        plt.show()
    except Exception as e:
        print(f"Confusion matrix plot failed: {e}")

# ────────────────────────── Experiment runner ─────────────────────────────────

def run_enhanced_experiment(ds_name, data, all_results):
    print(f"\n--- Starting Enhanced Experiment on {ds_name} dataset ---")
    X_df, y_sr = data["X"], data["y"]
    cat_cols = data["cat_features"]
    cont_cols = data["cont_features"]
    cat_cards = list(data["cat_cardinalities"].values())
    feat_names = cont_cols + cat_cols
    X_df = X_df[feat_names].copy()
    X_np, y_np = X_df.values, y_sr.values
    cont_idx = [feat_names.index(c) for c in cont_cols if c in feat_names]
    cat_idx = [feat_names.index(c) for c in cat_cols if c in feat_names]

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = {m: [] for m in ["KRAFT", "NoRules", "NoAttention", "NoUncertainty", "NoUncPenalty", "NoTemp", "ConcatRules",
                                    "SingleTreeRules", "MLP", "RF", "SVM", "XGBoost", "TabNet"]}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n==================== Fold {fold}/{N_SPLITS} ====================")
        # Leakage-free imputation per fold
        X_tr_raw, X_te_raw = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y_np[tr_idx], y_np[te_idx]
        imputer = KNNImputer(n_neighbors=5)
        X_tr_imp = imputer.fit_transform(pd.DataFrame(X_tr_raw, columns=feat_names))
        X_te_imp = imputer.transform(pd.DataFrame(X_te_raw, columns=feat_names))
        # Rules from fold-train only
        rules = get_enhanced_ensemble_rules(X_tr_imp, y_tr, feat_names)
        R_tr_full = apply_enhanced_rules(X_tr_imp, rules, feat_names)
        R_te = apply_enhanced_rules(X_te_imp, rules, feat_names)
        # Visualize rule coverage on first fold
        if fold == 1:
            print_human_rules(rules, feat_names)
            plot_rule_coverage(X_tr_imp, y_tr, rules, feat_names, ds_name, fold)
        # Scale continuous features (train-fit only)
        scaler = RobustScaler().fit(X_tr_imp[:, cont_idx]) if cont_idx else None
        X_tr_scaled = X_tr_imp.copy()
        X_te_scaled = X_te_imp.copy()
        if scaler is not None and cont_idx:
            X_tr_scaled[:, cont_idx] = scaler.transform(X_tr_imp[:, cont_idx])
            X_te_scaled[:, cont_idx] = scaler.transform(X_te_imp[:, cont_idx])
        # Train/validation split within fold-train (no leakage to test)
        X_trm, X_va, y_trm, y_va, R_trm, R_va = train_test_split(
            X_tr_scaled, y_tr, R_tr_full, test_size=0.2, stratify=y_tr, random_state=SEED
        )
        print(f"Pre-SMOTE class dist: {np.bincount(y_trm)}")
        pos_ratio = y_trm.sum() / len(y_trm)
        if pos_ratio < 0.3 or ds_name == "NAFLD":
            smote = SMOTE(random_state=SEED)
            X_combined = np.column_stack([X_trm, R_trm])
            X_combined_resampled, y_trm = smote.fit_resample(X_combined, y_trm)
            X_trm = X_combined_resampled[:, :X_trm.shape[1]]
            R_trm = X_combined_resampled[:, X_trm.shape[1]:]
            print(f"Post-SMOTE class dist: {np.bincount(y_trm)}")
        # Model variants (ablations)
        mods = {
            "KRAFT": {},
            "NoRules": {"no_rules": True},
            "NoAttention": {"no_attention": True},
            "NoUncertainty": {"no_uncertainty": True},
            "NoUncPenalty": {"no_unc_penalty": True},
            "NoTemp": {"no_temp": True},
            "ConcatRules": {"concat_rules": True},
            "SingleTreeRules": {},
        }
        # Per-variant training
        for mname, abl in mods.items():
            print(f"\n*** Training {mname} ***")
            rules_current = rules
            R_trm_current = R_trm
            R_va_current = R_va
            R_te_current = R_te
            if mname == "SingleTreeRules":
                rules_current = get_single_tree_rules(X_trm, y_trm, feat_names)
                R_trm_current = apply_enhanced_rules(X_trm, rules_current, feat_names)
                R_va_current = apply_enhanced_rules(X_va, rules_current, feat_names)
                R_te_current = apply_enhanced_rules(X_te_scaled, rules_current, feat_names)
            # Optuna with pruning
            study = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=SEED),
                                       pruner=optuna.pruners.MedianPruner(n_startup_trials=20))
            opt_data = dict(
                train=(X_trm[:, cont_idx] if cont_idx else np.empty((len(X_trm), 0)),
                       X_trm[:, cat_idx] if cat_idx else np.empty((len(X_trm), 0), dtype=int),
                       R_trm_current, y_trm),
                val=(X_va[:, cont_idx] if cont_idx else np.empty((len(X_va), 0)),
                     X_va[:, cat_idx] if cat_idx else np.empty((len(X_va), 0), dtype=int),
                     R_va_current, y_va),
                cat_cardinalities=cat_cards,
            )
            study.optimize(lambda t: enhanced_objective_for_optuna(t, opt_data), n_trials=N_TRIALS_OPTUNA, timeout=3600)
            best = study.best_params
            print(f"Optuna best params: {best}")
            mdl_par = {k: v for k, v in best.items() if k not in ("lr", "weight_decay", "rule_lambda")}
            opt_par = {k: best[k] for k in ("lr", "weight_decay")}
            rule_lambda = best.get("rule_lambda", 0.05)
            model = KRAFT(cat_cards, len(cont_idx), R_trm_current.shape[1], **mdl_par, run_ablations=abl).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), **opt_par)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=15, factor=0.5, min_lr=MIN_LEARNING_RATE)
            pos_weight = make_pos_weight(y_trm, DEVICE)
            criterion = BalancedFocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight, label_smoothing=0.05)
            def safe_tensor(arr):
                if arr.size == 0:
                    return torch.empty(0)
                cleaned = np.nan_to_num(arr.astype(np.float32), nan=0., posinf=1., neginf=-1.)
                return torch.from_numpy(np.clip(cleaned, -10, 10)).float()
            batch_size_dyn = min(BATCH_SIZE, len(y_trm) // 2 + 1)
            tr_loader = DataLoader(
                TensorDataset(
                    safe_tensor(X_trm[:, cont_idx] if cont_idx else np.empty((len(X_trm), 0))),
                    torch.from_numpy(X_trm[:, cat_idx] if cat_idx else np.empty((len(X_trm), 0), dtype=int)).long(),
                    safe_tensor(R_trm_current),
                    torch.from_numpy(y_trm).float(),
                ),
                batch_size=batch_size_dyn, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            va_loader = DataLoader(
                TensorDataset(
                    safe_tensor(X_va[:, cont_idx] if cont_idx else np.empty((len(X_va), 0))),
                    torch.from_numpy(X_va[:, cat_idx] if cat_idx else np.empty((len(X_va), 0), dtype=int)).long(),
                    safe_tensor(R_va_current),
                    torch.from_numpy(y_va).float(),
                ),
                batch_size=batch_size_dyn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            model, best_ep, _ = enhanced_train_and_evaluate(
                model, tr_loader, va_loader, optimizer, scheduler, criterion, PATIENCE, EPOCHS_FINAL, DEVICE, rule_lambda, prefix=mname
            )
            # Test evaluation with optional calibration (on validation set)
            te_loader = DataLoader(
                TensorDataset(
                    safe_tensor(X_te_scaled[:, cont_idx] if cont_idx else np.empty((len(X_te_scaled), 0))),
                    torch.from_numpy(X_te_scaled[:, cat_idx] if cat_idx else np.empty((len(X_te_scaled), 0), dtype=int)).long(),
                    safe_tensor(R_te_current),
                    torch.from_numpy(y_te).float(),
                ),
                batch_size=batch_size_dyn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )
            model.eval()
            preds, tgts, vars_list = [], [], []
            val_preds_for_cal, val_tgts_for_cal = [], []
            with torch.no_grad():
                # Collect validation preds to fit calibrator
                for xc, xcat, xr, tgt in va_loader:
                    logit, mean, log_var = model(xc.to(DEVICE), xcat.to(DEVICE), xr.to(DEVICE))
                    p = torch.sigmoid(logit).cpu().numpy().flatten()
                    val_preds_for_cal.extend(p)
                    val_tgts_for_cal.extend(tgt.numpy())
                for xc, xcat, xr, tgt in te_loader:
                    try:
                        logit, mean, log_var = model(xc.to(DEVICE), xcat.to(DEVICE), xr.to(DEVICE))
                        pred_probs = torch.sigmoid(logit).cpu().numpy()
                        valid_mask = np.isfinite(pred_probs).flatten()
                        if valid_mask.any():
                            preds.extend(pred_probs.flatten()[valid_mask])
                            tgts.extend(tgt.numpy()[valid_mask])
                            vars_list.extend(np.exp(log_var.cpu().numpy().flatten())[valid_mask])
                    except Exception:
                        continue
            if preds and tgts:
                preds_array = np.array(preds)
                tgts_array = np.array(tgts)
                # Post-hoc calibration (does not change AUC but improves calibration metrics)
                try:
                    preds_array = calibrate_probs(np.array(val_tgts_for_cal), np.array(val_preds_for_cal), preds_array, method='auto')
                except Exception as _:
                    pass
                pred_binary = (preds_array > 0.5).astype(int)
                metrics = calculate_enhanced_metrics(tgts_array, pred_binary, preds_array)
                fold_metrics[mname].append(metrics)
                print(f" {mname.upper()} TEST RESULTS")
                for k, v in metrics.items():
                    print(f"{k:<12} {v:.3f}")
                print(f"Optimal epochs: {best_ep}")
                if fold == 1 and mname in ["KRAFT", "NoRules", "ConcatRules"]:
                    plot_pr_curve(tgts_array, preds_array, ds_name, fold, mname)
                    plot_roc_curve(tgts_array, preds_array, ds_name, fold, mname)
                    plot_confusion_matrix(tgts_array, pred_binary, ds_name, fold, mname)
            else:
                print(f"No valid predictions for {mname}")
        # Baselines
        baseliners = dict(
            MLP=MLPClassifier(max_iter=1000, early_stopping=True, alpha=0.001, hidden_layer_sizes=(200, 100), random_state=SEED),
            RF=RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced_subsample", min_samples_split=10, max_features='sqrt', random_state=SEED),
            SVM=SVC(probability=True, class_weight="balanced", C=1.0, gamma='scale', random_state=SEED),
            XGBoost=XGBClassifier(use_label_encoder=False, eval_metric="logloss", early_stopping_rounds=10, max_depth=8, learning_rate=0.05, n_estimators=500, random_state=SEED),
            TabNet=TabNetClassifier(verbose=0, seed=SEED, n_steps=5, gamma=1.5),
        )
        for bname, clf in baseliners.items():
            print(f"\n*** Evaluating Baseline: {bname} ***")
            try:
                if bname == 'XGBoost':
                    clf.fit(X_trm, y_trm, eval_set=[(X_va, y_va)], verbose=False)
                elif bname == 'TabNet':
                    clf.fit(X_trm, y_trm, eval_set=[(X_va, y_va)], patience=20)
                else:
                    clf.fit(X_trm, y_trm)
                prob_test = clf.predict_proba(X_te_scaled)[:, 1]
                pred_test = (prob_test > 0.5).astype(int)
                metrics = calculate_enhanced_metrics(y_te, pred_test, prob_test)
                fold_metrics[bname].append(metrics)
                print(f" {bname.upper()} BASELINE RESULTS")
                for k, v in metrics.items():
                    print(f"{k:<12} {v:.3f}")
            except Exception as e:
                print(f"Baseline {bname} failed: {e}")
                prob_test = np.full(len(y_te), 0.5)
                pred_test = np.zeros(len(y_te), dtype=int)
                metrics = calculate_enhanced_metrics(y_te, pred_test, prob_test)
                fold_metrics[bname].append(metrics)

    # Summary with CIs and FDR-corrected p-values
    print(f"\n=== RESULTS SUMMARY FOR {ds_name} ===")
    def compute_ci(data, confidence=0.95):
        if len(data) < 2:
            return np.mean(data) if len(data) else float('nan'), np.nan, np.nan
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        t_crit = student_t.ppf((1 + confidence) / 2.0, n - 1)
        margin = t_crit * std / np.sqrt(n)
        return mean, mean - margin, mean + margin
    avg_dict = {}
    for m in fold_metrics.keys():
        if fold_metrics[m]:
            dfm = pd.DataFrame(fold_metrics[m])
            avg_dict[m] = {metric: f"{dfm[metric].mean():.4f} ± {dfm[metric].std():.4f}" for metric in dfm.columns}
    avg_df = pd.DataFrame(avg_dict).T
    main_model = "KRAFT"
    key_metrics = ['AUC', 'F1', 'BalancedAcc']
    print("\nPerformance with FDR-corrected p-values (vs. KRAFT):")
    for metric in key_metrics:
        main_vals = [fold[metric] for fold in fold_metrics[main_model]] if fold_metrics[main_model] else []
        pvals = []
        models = []
        for m in fold_metrics.keys():
            if m == main_model or not fold_metrics[m]:
                continue
            other_vals = [fold[metric] for fold in fold_metrics[m]]
            p = perform_wilcoxon_test(main_vals, other_vals)
            pvals.append(p)
            models.append(m)
        if pvals:
            adj = fdr_bh(pvals)
            print(f"\n--- {metric} ---")
            for mdl, p_raw, p_adj in zip(models, pvals, adj):
                avg_str = avg_df.loc[mdl, metric] if (mdl in avg_df.index and metric in avg_df.columns) else "N/A"
                sig = "*" if (not np.isnan(p_adj) and p_adj < 0.05) else ""
                print(f"{mdl:<16} {avg_str:<18} p={p_raw:.4f} | FDR={p_adj:.4f}{sig}")
    print("\nFull Average Metrics Table:")
    print(avg_df.to_string().replace("KRAFT", "**KRAFT**"))
    print("\nNote: p-values are BH-FDR corrected across models per metric. Metrics are mean ± std across folds.")

# ───────────────────────── Data processing (unchanged) ────────────────────────

def validate_and_process(datasets):
    processed = {}
    _ID_RE = re.compile(
        r"^(patient[_\s]?id|case[_\s]?no\.?|id|record[_\s]?number)$|" r"(?:\b(?:id|no\.|number)\b$)",
        flags=re.I | re.VERBOSE,
    )
    label_map = {
        "GDM": "Class Label(GDM /Non GDM)",
        "NAFLD": "Type of Disease (Mild illness=1, Severe illness=2)",
        "Diabetes": "CLASS",
        "Cardio": "target",
        "PIMA": "Outcome",
    }
    for name, df in datasets.items():
        print(f"\n{name} Data Validation:")
        print(f" Shape: {df.shape}, Missing total: {df.isnull().sum().sum()}")
        label_col = label_map.get(name)
        if not label_col or label_col not in df.columns:
            print(f" ✗ label column not found; skipping {name}")
            continue
        print(f" Label dist ({label_col}):")
        print(df[label_col].value_counts())
        df.columns = (
            df.columns.str.strip().str.replace("\u00A0", " ").str.replace("\r", " ")
        )
        dropped = [c for c in df.columns if _ID_RE.search(c.lower())]
        print(f" Dropping columns: {set(dropped)}")
        df = df.drop(columns=dropped, errors="ignore")
        unique_labels = sorted(df[label_col].unique())
        if unique_labels == [1, 2]:
            df[label_col] = df[label_col].map({1: 0, 2: 1})
            print(f"\nConverting label in {name} from [1, 2] to [0, 1]")
        X, y = df.drop(columns=[label_col]), df[label_col]
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        cont_cols = [c for c in X.select_dtypes(include=np.number).columns if c not in cat_cols]
        cat_card = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            card = len(le.classes_)
            if card > 100:
                print(f"Warning: {col} has cardinality {card} >100; capping at 100")
            cat_card[col] = min(card, 100)
        processed[name] = dict(
            X=X, y=y, cat_features=cat_cols, cont_features=cont_cols, cat_cardinalities=cat_card
        )
    return processed

# ───────────────────────────── Main entrypoint ────────────────────────────────

def enhanced_main():
    print("=== KRAFT: Enhanced Stable RuleTransNet ===")
    print(f"Device: {DEVICE}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print("\nLoading and processing datasets...")
    try:
        gdm = pd.read_csv("KaggleGDM.csv")
        nafld = pd.read_excel("NAFLD.xlsx")
        diabetes = pd.read_csv("MendeleyDiabetes.csv")
        cardio = pd.read_csv("CardioMendeley.csv")
        pima = pd.read_csv("PIMA.csv")
        if "CLASS" in diabetes.columns:
            diabetes["CLASS"] = (
                diabetes["CLASS"].astype(str).str.strip().map({"Y": 1, "P": 1, "N": 0})
            )
        if "Outcome" not in pima.columns:
            pima.columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
                "Outcome",
            ]
    except Exception as e:
        print(f"FATAL: dataset load failure\n{e}")
        raise
    dfs = dict(GDM=gdm, NAFLD=nafld, Diabetes=diabetes, Cardio=cardio, PIMA=pima)
    for name in ['GDM', 'NAFLD', 'Diabetes', 'Cardio', 'PIMA']:
        print(f"{name} shape: {dfs[name].shape}")
    processed = validate_and_process(dfs)
    exp_datasets = ["Diabetes", "Cardio", "GDM", "NAFLD"]
    model_names = [
        "KRAFT",
        "NoRules",
        "NoAttention",
        "NoUncertainty",
        "NoUncPenalty",
        "NoTemp",
        "ConcatRules",
        "SingleTreeRules",
        "MLP",
        "RF",
        "SVM",
        "XGBoost",
        "TabNet",
    ]
    all_results = {ds: {m: [] for m in model_names} for ds in exp_datasets}
    for ds in exp_datasets:
        if ds in processed:
            run_enhanced_experiment(ds, processed[ds], all_results)
    print("\n\n========== FINAL RESULTS ==========")
    for ds, res in all_results.items():
        print(f"\n=== {ds} RESULTS ===")
        if res and any(res.values()):
            try:
                avg = {m: {k: np.mean([d[k] for d in folds]) for k in folds[0]} for m, folds in res.items() if folds}
                results_df = pd.DataFrame(avg).T.round(4)
                print(results_df)
            except Exception:
                print("Summary failed, partial results only")
        else:
            print("No valid results for this dataset")
    print("\n=== EXECUTION COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    t0 = time.time()
    print(f"Execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        enhanced_main()
        print(f"\nTotal execution time: {(time.time() - t0) / 60:.2f} minutes")
        print(f"Peak RAM usage: {psutil.Process(os.getpid()).memory_info().rss / 1e6:.1f} MB")
    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
    print("=== KRAFT RUN COMPLETED ===")