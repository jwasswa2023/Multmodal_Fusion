# ==========================================
# CODE A
# TRUE LATE FUSION:
#   RDKit-LGBM + Mol2Vec-LGBM + AttentiveFP + SMILES-BiGRU
#   → META LGBM (prediction-level fusion)
#   → 3-seed evaluation + ensemble predictions for Code B
# ==========================================
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl  # required by AttentiveFP (DGL backend)

from rdkit import Chem

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer, RDKitDescriptors
from deepchem.data import NumpyDataset
from deepchem.models import AttentiveFPModel

from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# =========================================================
# 0) Helpers
# =========================================================
def summarize(values, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mu = arr.mean()
    sd = arr.std(ddof=1) if n > 1 else 0.0
    if n > 1:
        tcrit = 4.303 if n == 3 else 1.96
        margin = tcrit * sd / math.sqrt(n)
    else:
        margin = 0.0
    return mu, sd, mu - margin, mu + margin

def stratified_split_indices(y, test_size=0.2, n_bins=5, random_state=0):
    y = np.asarray(y).reshape(-1)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(y, qs))
    idx_all = np.arange(len(y))
    if len(bins) <= 2:
        print("[Warn] Stratification collapsed; using unstratified split.")
        y_binned = np.zeros_like(y, dtype=int)
    else:
        y_binned = np.digitize(y, bins[1:-1], right=True)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(sss.split(idx_all, y_binned))
    return train_idx, test_idx

# ---------- SMILES encoder (char-level BiGRU) ----------
class SmilesDataset(Dataset):
    def __init__(self, X_ids, y):
        self.X = torch.from_numpy(X_ids.astype(np.int64))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SmilesEncoderRegressor(nn.Module):
    """
    Simple BiGRU → hidden → scalar regressor.
    """
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, pad_idx=0, p_drop=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)              # (B, L, E)
        out, h = self.gru(emb)               # h: (2, B, H)
        h_fw = h[0]
        h_bw = h[1]
        h_cat = torch.cat([h_fw, h_bw], dim=-1)  # (B, 2H)
        h_cat = self.dropout(h_cat)
        pred = self.fc(h_cat).squeeze(-1)    # (B,)
        return pred

def build_smiles_vocab(smiles_train, max_len_cap=200):
    special = ["<pad>", "<unk>"]
    chars = sorted({ch for s in smiles_train for ch in s})
    itos = special + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    max_len = min(max_len_cap, max(len(s) for s in smiles_train))
    return stoi, itos, max_len

def encode_smiles_array(smiles, stoi, max_len):
    pad_idx = stoi["<pad>"]; unk_idx = stoi["<unk>"]
    out = np.zeros((len(smiles), max_len), dtype=np.int64)
    for i, s in enumerate(smiles):
        ids = [stoi.get(ch, unk_idx) for ch in s[:max_len]]
        if len(ids) < max_len:
            ids += [pad_idx] * (max_len - len(ids))
        out[i, :] = np.array(ids, dtype=np.int64)
    return out

def train_smiles_model_and_predict(
    train_loader_train,
    train_loader_eval,
    test_loader_eval,
    device,
    seed=0,
    n_epochs=20,
    vocab_size=None,
    pad_idx=0,
    emb_dim=64,
    hidden_dim=128,
    lr=1e-3
):
    """
    Train SMILES BiGRU regressor on train_loader_train, then
    return predictions on train_loader_eval & test_loader_eval.
    Predictions are aligned with dataset order in the *_eval loaders.
    """
    torch.manual_seed(seed)
    model = SmilesEncoderRegressor(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        pad_idx=pad_idx,
        p_drop=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Train ----
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optim.step()

    # ---- Predict helper ----
    @torch.no_grad()
    def predict_from_loader(loader):
        model.eval()
        preds_all = []
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            preds_all.append(out.cpu().numpy())
        return np.concatenate(preds_all, axis=0)

    y_train_pred = predict_from_loader(train_loader_eval)
    y_test_pred  = predict_from_loader(test_loader_eval)
    return y_train_pred, y_test_pred

def train_attentivefp_and_predict(train_dataset, test_dataset, seed=0):
    """
    Train AttentiveFP on graphs only → scalar predictions (unimodal).
    """
    gnn = AttentiveFPModel(
        n_tasks=1,
        mode="regression",
        batch_normalize=True,
        random_seed=seed,
        model_dir=f"attfp_unimodal_seed{seed}"
    )
    print(f"[Info] Training AttentiveFP unimodal (seed={seed})...")
    gnn.fit(train_dataset, nb_epoch=50)

    y_train_pred = np.asarray(gnn.predict(train_dataset)).reshape(-1)
    y_test_pred  = np.asarray(gnn.predict(test_dataset)).reshape(-1)
    print(f"[Info] AttentiveFP preds (seed={seed}) — Train: {y_train_pred.shape}, Test: {y_test_pred.shape}")
    return y_train_pred, y_test_pred

# =========================================================
# 1) Load Mol2Vec + RDKit + targets + SMILES
# =========================================================
data = np.load("/content/mol2vec_rdkit_features.npz", allow_pickle=True)
X_m2v = data["X_mol2vec"]      # (N, d_m2v)
X_rd  = data["X_rdkit"]        # (N, d_rd)
y      = data["y"].astype(float)
smiles = data["smiles"].astype(str)

print("[Info] Loaded NPZ shapes:")
print("  Mol2Vec:", X_m2v.shape)
print("  RDKit  :", X_rd.shape)
print("  y      :", y.shape)
print("  SMILES :", smiles.shape)

N = len(y)
assert X_m2v.shape[0] == N and X_rd.shape[0] == N and smiles.shape[0] == N

# =========================================================
# 2) Build AttentiveFP graphs (DGL backend)
# =========================================================
graph_featurizer = MolGraphConvFeaturizer(use_edges=True)
X_graph = graph_featurizer.featurize(smiles.tolist())
mask = np.array([g is not None for g in X_graph])
if not mask.all():
    print(f"[Info] Dropping {np.sum(~mask)} molecules failing graph featurization.")

X_graph = np.asarray([g for g, ok in zip(X_graph, mask) if ok], dtype=object)
X_m2v   = X_m2v[mask]
X_rd    = X_rd[mask]
y       = y[mask]
smiles  = smiles[mask]

print("[Info] After graph alignment:")
print("  N       :", len(y))
print("  Mol2Vec :", X_m2v.shape)
print("  RDKit   :", X_rd.shape)
print("  Graph   :", X_graph.shape)

# =========================================================
# 3) Single stratified split (fixed)
# =========================================================
train_idx, test_idx = stratified_split_indices(
    y, test_size=0.2, n_bins=5, random_state=0
)
print(f"[Info] Stratified split — Train: {len(train_idx)}, Test: {len(test_idx)}")

X_train_graph = X_graph[train_idx]
X_test_graph  = X_graph[test_idx]
X_train_m2v   = X_m2v[train_idx]
X_test_m2v    = X_m2v[test_idx]
X_train_rd    = X_rd[train_idx]
X_test_rd     = X_rd[test_idx]
y_train       = y[train_idx]
y_test        = y[test_idx]
smiles_train  = smiles[train_idx]
smiles_test   = smiles[test_idx]

train_dataset_graph = NumpyDataset(X_train_graph, y_train.reshape(-1, 1))
test_dataset_graph  = NumpyDataset(X_test_graph,  y_test.reshape(-1, 1))

# =========================================================
# 4) SMILES tokenization and DataLoaders
# =========================================================
stoi, itos, max_len = build_smiles_vocab(smiles_train, max_len_cap=200)
X_train_seq_ids = encode_smiles_array(smiles_train, stoi, max_len)
X_test_seq_ids  = encode_smiles_array(smiles_test,  stoi, max_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds_seq = SmilesDataset(X_train_seq_ids, y_train)
test_ds_seq  = SmilesDataset(X_test_seq_ids,  y_test)

# loader for training (shuffle=True)
train_loader_seq_train = DataLoader(train_ds_seq, batch_size=32, shuffle=True)
# loaders for evaluation (no shuffle, to preserve order)
train_loader_seq_eval  = DataLoader(train_ds_seq, batch_size=64, shuffle=False)
test_loader_seq_eval   = DataLoader(test_ds_seq,  batch_size=64, shuffle=False)

# =========================================================
# 5) Hyperparameter grid for unimodal LGBMs (Mol2Vec-only, RDKit-only)
# =========================================================
param_distributions_lgbm = {
    "verbose": [-1],
    "boosting_type": ["gbdt"],
    "num_leaves": [5, 15, 30],
    "max_depth": [50, 100, 300, -1],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample_for_bin": [50, 100, 200],
    "min_split_gain": [0.0],
    "min_child_weight": [0.001],
    "min_child_samples": [20],
    "subsample": [0.7, 0.8, 1.0],
    "feature_fraction": [0.7, 0.8, 1.0],
}

print("[Info] Tuning RDKit-only LGBM (seed=0)...")
base_lgbm_rd = LGBMRegressor()
rs_rd = RandomizedSearchCV(
    estimator=base_lgbm_rd,
    param_distributions=param_distributions_lgbm,
    n_iter=60,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=0,
)
rs_rd.fit(X_train_rd, y_train)
best_params_rd = rs_rd.best_params_

print("[Info] Best RDKit LGBM params:")
for k, v in best_params_rd.items():
    print(f"  {k}: {v}")

print("[Info] Tuning Mol2Vec-only LGBM (seed=0)...")
base_lgbm_m2v = LGBMRegressor()
rs_m2v = RandomizedSearchCV(
    estimator=base_lgbm_m2v,
    param_distributions=param_distributions_lgbm,
    n_iter=60,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=0,
)
rs_m2v.fit(X_train_m2v, y_train)
best_params_m2v = rs_m2v.best_params_

print("[Info] Best Mol2Vec LGBM params:")
for k, v in best_params_m2v.items():
    print(f"  {k}: {v}")

# =========================================================
# 6) Base model predictions (seed=0) for META HP tuning
#    TRUE LATE FUSION: use PREDICTIONS, not embeddings
# =========================================================
print("[Info] Preparing base-model predictions for META LGBM HP tuning (seed=0)...")

# ---- AttentiveFP unimodal (graphs only) ----
y_train_att_0, y_test_att_0 = train_attentivefp_and_predict(
    train_dataset_graph, test_dataset_graph, seed=0
)

# ---- SMILES BiGRU unimodal ----
y_train_smiles_0, y_test_smiles_0 = train_smiles_model_and_predict(
    train_loader_seq_train,
    train_loader_seq_eval,
    test_loader_seq_eval,
    device=device,
    seed=0,
    n_epochs=20,
    vocab_size=len(itos),
    pad_idx=stoi["<pad>"],
    emb_dim=64,
    hidden_dim=128,
    lr=1e-3,
)

# ---- RDKit-only LGBM ----
params_rd_0 = best_params_rd.copy()
params_rd_0["random_state"] = 0
rd_model_0 = LGBMRegressor(**params_rd_0)
rd_model_0.fit(X_train_rd, y_train)
y_train_rd_0 = rd_model_0.predict(X_train_rd)
y_test_rd_0  = rd_model_0.predict(X_test_rd)

# ---- Mol2Vec-only LGBM ----
params_m2v_0 = best_params_m2v.copy()
params_m2v_0["random_state"] = 0
m2v_model_0 = LGBMRegressor(**params_m2v_0)
m2v_model_0.fit(X_train_m2v, y_train)
y_train_m2v_0 = m2v_model_0.predict(X_train_m2v)
y_test_m2v_0  = m2v_model_0.predict(X_test_m2v)

# ---- META FEATURES = stacked predictions from 4 base models ----
# Order: [RDKit_pred, Mol2Vec_pred, AttentiveFP_pred, SMILES_pred]
X_train_meta_0 = np.column_stack(
    [y_train_rd_0, y_train_m2v_0, y_train_att_0, y_train_smiles_0]
)
X_test_meta_0 = np.column_stack(
    [y_test_rd_0, y_test_m2v_0, y_test_att_0, y_test_smiles_0]
)

print("[Info] META feature dims (seed=0) — Train:", X_train_meta_0.shape,
      "Test:", X_test_meta_0.shape)

# =========================================================
# 7) Hyperparameter search for META LGBM (late fusion)
# =========================================================
print("[Info] META LGBM hyperparameter search on base predictions (TRUE LATE FUSION)...")

base_lgbm_meta = LGBMRegressor()
rs_meta = RandomizedSearchCV(
    estimator=base_lgbm_meta,
    param_distributions=param_distributions_lgbm,
    n_iter=60,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0,
    random_state=0,
)
rs_meta.fit(X_train_meta_0, y_train)
best_params_meta = rs_meta.best_params_

print("[Info] Best META LGBM params:")
for k, v in best_params_meta.items():
    print(f"  {k}: {v}")

# =========================================================
# 8) 3-seed evaluation: TRUE LATE FUSION
#    + store ensemble predictions for Code B
# =========================================================
seeds = [0, 1, 2]
r2_list, rmse_list, mae_list = [], [], []

ensemble_preds_meta_lgbm = []   # list of (n_test,) predictions
baseline_pred_meta_lgbm = None  # seed=0 predictions

print("\n" + "="*120)
print("TRUE LATE FUSION — RDKit-LGBM + Mol2Vec-LGBM + AttentiveFP + SMILES-BiGRU → META LGBM (prediction-level)")
print("="*120)

for i, seed in enumerate(seeds):
    print(f"\n=== Seed {seed} ===")

    # ---- AttentiveFP unimodal ----
    y_train_att, y_test_att = train_attentivefp_and_predict(
        train_dataset_graph, test_dataset_graph, seed=seed
    )

    # ---- SMILES unimodal ----
    y_train_smiles, y_test_smiles = train_smiles_model_and_predict(
        train_loader_seq_train,
        train_loader_seq_eval,
        test_loader_seq_eval,
        device=device,
        seed=seed,
        n_epochs=20,
        vocab_size=len(itos),
        pad_idx=stoi["<pad>"],
        emb_dim=64,
        hidden_dim=128,
        lr=1e-3,
    )

    # ---- RDKit-only LGBM ----
    params_rd_seed = best_params_rd.copy()
    params_rd_seed["random_state"] = seed
    rd_model = LGBMRegressor(**params_rd_seed)
    rd_model.fit(X_train_rd, y_train)
    y_train_rd = rd_model.predict(X_train_rd)
    y_test_rd  = rd_model.predict(X_test_rd)

    # ---- Mol2Vec-only LGBM ----
    params_m2v_seed = best_params_m2v.copy()
    params_m2v_seed["random_state"] = seed
    m2v_model = LGBMRegressor(**params_m2v_seed)
    m2v_model.fit(X_train_m2v, y_train)
    y_train_m2v = m2v_model.predict(X_train_m2v)
    y_test_m2v  = m2v_model.predict(X_test_m2v)

    # ---- Build META features from 4 base predictions ----
    X_train_meta = np.column_stack(
        [y_train_rd, y_train_m2v, y_train_att, y_train_smiles]
    )
    X_test_meta = np.column_stack(
        [y_test_rd, y_test_m2v, y_test_att, y_test_smiles]
    )

    # ---- META LGBM ----
    params_meta_seed = best_params_meta.copy()
    params_meta_seed["random_state"] = seed
    meta_model = LGBMRegressor(**params_meta_seed)
    meta_model.fit(X_train_meta, y_train)

    y_pred = meta_model.predict(X_test_meta)

    # store predictions for Code B
    ensemble_preds_meta_lgbm.append(y_pred.copy())
    if i == 0:
        baseline_pred_meta_lgbm = y_pred.copy()

    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"R2   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")

    r2_list.append(r2)
    rmse_list.append(rmse)
    mae_list.append(mae)

# =========================================================
# 9) Summary
# =========================================================
r2_mean, r2_std, r2_lo, r2_hi = summarize(r2_list)
rmse_mean, rmse_std, rmse_lo, rmse_hi = summarize(rmse_list)
mae_mean, mae_std, mae_lo, mae_hi = summarize(mae_list)

print("\n=== Summary: TRUE LATE FUSION (RDKit + Mol2Vec + AttentiveFP + SMILES → META LGBM) ===")
print(f"R2   : {r2_mean:.4f} ± {r2_std:.4f}, 95% CI=({r2_lo:.4f}, {r2_hi:.4f})")
print(f"RMSE : {rmse_mean:.4f} ± {rmse_std:.4f}, 95% CI=({rmse_lo:.4f}, {rmse_hi:.4f})")
print(f"MAE  : {mae_mean:.4f} ± {mae_std:.4f}, 95% CI=({mae_lo:.4f}, {mae_hi:.4f})")

# For CODE B, you now have:
#   - y_test
#   - ensemble_preds_meta_lgbm  (list of 3 arrays, shape (n_test,))
#   - baseline_pred_meta_lgbm   (array, shape (n_test,))

