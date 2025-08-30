# -*- coding: utf-8 -*-
import os
import re
import json
import copy
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- Classification metrics / models ----
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ---- Regression metrics / models ----
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

# ---- Anomaly metrics / models ----
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ---- PyTorch ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# 0) Public: Early Stopper
# =============================================================================
@dataclass
class EarlyStopper:
    mode: str = "max"          # "max" for acc/R2/AUC; "min" for loss/RMSE
    patience: int = 10
    best_metric: Optional[float] = None
    num_bad_epochs: int = 0
    best_state_dict: Optional[dict] = None

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        return (metric > self.best_metric) if self.mode == "max" else (metric < self.best_metric)

    def step(self, metric: float, model: nn.Module) -> bool:
        improved = self._is_better(metric)
        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.num_bad_epochs += 1
        return improved

    def should_stop(self) -> bool:
        return self.num_bad_epochs > self.patience

    def load_best(self, model: nn.Module, device: torch.device):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            model.to(device)

# =============================================================================
# 1) Classification models (CNN1D / Transformer)
# =============================================================================
class CNN1DClassifier(nn.Module):
    def __init__(self, seq_len, n_classes, in_channels=1, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> (batch, 1, seq_len)
        out = self.conv(x)
        out = self.classifier(out)
        return out  # logits (batch, n_classes)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len, n_classes, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # -> (batch, seq_len, 1)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits

# ======================= Classification Pipeline (ML=GridSearch on train; NN uses val early stop) =======================
class ClassificationModelPipeline:
    def __init__(self, train_data, test_data, val_data=None,
                 val_size=0.2, val_random_state=42, early_stop_patience=15, cv_folds=5):
        self.X_train_full = np.array([d["x"] for d in train_data], dtype=np.float32)
        self.y_train_full = np.array([d["label"] for d in train_data])
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([d["label"] for d in test_data])

        # If val_data is not provided, split from training set (stratified), only used by NN
        if val_data is None:
            self.X_tr, self.X_val, y_tr, y_val = train_test_split(
                self.X_train_full, self.y_train_full, test_size=val_size,
                stratify=self.y_train_full, random_state=val_random_state
            )
        else:
            self.X_tr, y_tr = self.X_train_full, self.y_train_full
            self.X_val = np.array([d["x"] for d in val_data], dtype=np.float32)
            y_val = np.array([d["label"] for d in val_data])

        # Label mapping uses union of train/val/test
        all_labels = sorted(set(np.concatenate([y_tr, y_val, self.y_test])))
        self.l2i = {l: i for i, l in enumerate(all_labels)}
        self.y_tr_i   = np.array([self.l2i[y] for y in y_tr], dtype=np.int64)
        self.y_val_i  = np.array([self.l2i[y] for y in y_val], dtype=np.int64)
        self.y_test_i = np.array([self.l2i[y] for y in self.y_test], dtype=np.int64)

        self.early_stop_patience = early_stop_patience
        self.cv_folds = cv_folds

    def _grid_search_classical_ml(self, X_train_flat, y_train_i):
        """Perform grid search on training set and return a dictionary of best fitted models."""
        results = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # SVC (with standardization)
        pipe_svc = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
        param_svc = {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10, 100],
            "svc__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
        gs_svc = GridSearchCV(pipe_svc, param_grid=param_svc, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
        gs_svc.fit(X_train_flat, y_train_i)
        results["SVM"] = gs_svc

        # KNN (with standardization)
        pipe_knn = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
        param_knn = {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2]
        }
        gs_knn = GridSearchCV(pipe_knn, param_grid=param_knn, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
        gs_knn.fit(X_train_flat, y_train_i)
        results["KNN"] = gs_knn

        # RandomForest (no standardization needed)
        rf = RandomForestClassifier(random_state=42)
        param_rf = {
            "n_estimators": [100, 300, 600],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
        gs_rf = GridSearchCV(rf, param_grid=param_rf, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
        gs_rf.fit(X_train_flat, y_train_i)
        results["RandomForest"] = gs_rf

        return results

    def train_and_evaluate(self, nn_epochs=100, batch_size=32):
        results = {}

        # ---------------- Traditional ML: GridSearchCV on train only, evaluate on test ----------------
        flat_tr  = self.X_tr.reshape(len(self.X_tr), -1)
        flat_te  = self.X_test.reshape(len(self.X_test), -1)

        best_ml = self._grid_search_classical_ml(flat_tr, self.y_tr_i)
        for name, gs in best_ml.items():
            pred = gs.predict(flat_te)
            results[name] = accuracy_score(self.y_test_i, pred)
            print(f"[ML][{name}] best_params={gs.best_params_}, CV_best_acc={gs.best_score_:.4f}, Test_acc={results[name]:.4f}")

        # ---------------- Deep Learning: train on train, early stop on val, evaluate on test ----------------
        # ONLY fit scaler on train
        x_scaler = StandardScaler()
        flat_val = self.X_val.reshape(len(self.X_val), -1)
        flat_tr_s  = x_scaler.fit_transform(flat_tr)
        flat_val_s = x_scaler.transform(flat_val)
        flat_te_s  = x_scaler.transform(flat_te)

        X_tr_s  = flat_tr_s.reshape(self.X_tr.shape).astype(np.float32)
        X_val_s = flat_val_s.reshape(self.X_val.shape).astype(np.float32)
        X_te_s  = flat_te_s.reshape(self.X_test.shape).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds_tr  = TensorDataset(torch.from_numpy(X_tr_s),  torch.from_numpy(self.y_tr_i).long())
        ds_val = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(self.y_val_i).long())
        loader_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=False)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

        def _fit_cls(model, epochs=nn_epochs):
            model = model.to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
            loss_fn = nn.CrossEntropyLoss()

            stopper = EarlyStopper(mode="max", patience=self.early_stop_patience)

            for ep in range(1, epochs + 1):
                model.train()
                for xb, yb in loader_tr:
                    xb = xb.to(device).float()
                    yb = yb.to(device).long()
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                scheduler.step()

                # Validation
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in loader_val:
                        xb = xb.to(device).float()
                        yb = yb.to(device).long()
                        logits = model(xb)
                        pred = logits.argmax(dim=1)
                        correct += (pred == yb).sum().item()
                        total += yb.numel()
                val_acc = correct / max(1, total)

                stopper.step(val_acc, model)
                if stopper.should_stop():
                    print(f"[NN-CLS] Early stopped at epoch {ep}, best val_acc={stopper.best_metric:.4f}")
                    break

            stopper.load_best(model, device)

            model.eval()
            with torch.no_grad():
                xt = torch.from_numpy(X_te_s).to(device).float()
                logits = model(xt).cpu()
                preds = logits.argmax(dim=1).numpy()
            return preds

        cnn = CNN1DClassifier(seq_len=self.X_tr.shape[1], n_classes=len(self.l2i))
        pred = _fit_cls(cnn, epochs=nn_epochs)
        results["CNN1D"] = accuracy_score(self.y_test_i, pred)

        tfm = TransformerClassifier(seq_len=self.X_tr.shape[1], n_classes=len(self.l2i))
        pred = _fit_cls(tfm, epochs=nn_epochs)
        results["Transformer"] = accuracy_score(self.y_test_i, pred)

        for k, v in results.items():
            print(f"{k}: Accuracy = {v:.2%}")
        return results

# =============================================================================
# 2) Regression models (CNN1D / Transformer)
# =============================================================================
class CNN1DRegressor(nn.Module):
    def __init__(self, seq_len, in_channels=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.head(out)
        return out.squeeze(-1)

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)

# ======================= Regression Pipeline (ML=GridSearch on train; NN uses val early stop) =======================
class RegressionModelPipeline:
    def __init__(self, train_data, test_data, val_data=None,
                 val_size=0.2, val_random_state=42, early_stop_patience=10, cv_folds=5):
        self.X_train_full = np.array([d["x"] for d in train_data], dtype=np.float32)
        self.y_train_full = np.array([d["y"] for d in train_data], dtype=np.float32)
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([d["y"] for d in test_data], dtype=np.float32)

        if val_data is None:
            self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
                self.X_train_full, self.y_train_full, test_size=val_size,
                random_state=val_random_state
            )
        else:
            self.X_tr, self.y_tr = self.X_train_full, self.y_train_full
            self.X_val = np.array([d["x"] for d in val_data], dtype=np.float32)
            self.y_val = np.array([d["y"] for d in val_data], dtype=np.float32)

        self.early_stop_patience = early_stop_patience
        self.cv_folds = cv_folds

    def _grid_search_regression_ml(self, X_train_flat, y_train):
        """Perform grid search for regression models on training set and return a dictionary of best fitted models."""
        results = {}
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # LinearRegression (can simply search whether to include intercept)
        pipe_lr = Pipeline([("lr", LinearRegression())])
        param_lr = {"lr__fit_intercept": [True, False]}
        gs_lr = GridSearchCV(pipe_lr, param_grid=param_lr, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs_lr.fit(X_train_flat, y_train)
        results["LinearRegression"] = gs_lr

        # SVR (with standardization)
        pipe_svr = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
        param_svr = {
            "svr__kernel": ["rbf"],
            "svr__C": [0.1, 1, 10, 100],
            "svr__epsilon": [0.01, 0.1, 0.5, 1.0],
            "svr__gamma": ["scale", "auto", 0.01, 0.1, 1]
        }
        gs_svr = GridSearchCV(pipe_svr, param_grid=param_svr, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs_svr.fit(X_train_flat, y_train)
        results["SVR"] = gs_svr

        # PLSRegression (with standardization; n_components limited by samples/features)
        max_components = max(1, min(10, X_train_flat.shape[1], len(y_train) - 1))
        pipe_pls = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
        param_pls = {"pls__n_components": list(range(1, max_components + 1))}
        gs_pls = GridSearchCV(pipe_pls, param_grid=param_pls, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs_pls.fit(X_train_flat, y_train)
        results["PLSR"] = gs_pls

        # RandomForestRegressor
        rf = RandomForestRegressor(random_state=42)
        param_rf = {
            "n_estimators": [200, 500, 800],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
        gs_rf = GridSearchCV(rf, param_grid=param_rf, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=0)
        gs_rf.fit(X_train_flat, y_train)
        results["RandomForestRegressor"] = gs_rf

        return results

    def train_and_evaluate(self, nn_epochs=100, batch_size=32):
        results = {}

        # ---------------- Traditional ML: GridSearchCV on train only, evaluate on test ----------------
        flat_tr = self.X_tr.reshape(len(self.X_tr), -1)
        flat_te = self.X_test.reshape(len(self.X_test), -1)

        best_ml = self._grid_search_regression_ml(flat_tr, self.y_tr)
        for name, gs in best_ml.items():
            pred = gs.predict(flat_te)
            r2  = r2_score(self.y_test, pred)
            rmse = root_mean_squared_error(self.y_test, pred)
            results[name] = {"r2": r2, "rmse": rmse}
            print(f"[ML][{name}] best_params={gs.best_params_}, CV_best_negRMSE={gs.best_score_:.4f}, Test_R2={r2:.4f}, Test_RMSE={rmse:.4f}")

        # ---------------- Deep Learning: train on train, early stop on val, evaluate on test ----------------
        x_scaler = StandardScaler()
        flat_tr_s  = x_scaler.fit_transform(flat_tr)      # Fit only on train
        flat_val_s = x_scaler.transform(self.X_val.reshape(len(self.X_val), -1))
        flat_te_s  = x_scaler.transform(flat_te)

        X_tr_s  = flat_tr_s.reshape(self.X_tr.shape).astype(np.float32)
        X_val_s = flat_val_s.reshape(self.X_val.shape).astype(np.float32)
        X_te_s  = flat_te_s.reshape(self.X_test.shape).astype(np.float32)

        y_scaler = StandardScaler()
        y_tr_s = y_scaler.fit_transform(self.y_tr.reshape(-1, 1)).reshape(-1).astype(np.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ds_tr  = TensorDataset(torch.from_numpy(X_tr_s),  torch.from_numpy(y_tr_s))
        ds_val = TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(self.y_val))
        loader_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=False)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

        def _fit_reg(model, epochs=nn_epochs):
            model = model.to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.5)
            loss_fn = nn.MSELoss()

            stopper = EarlyStopper(mode="min", patience=self.early_stop_patience)

            for ep in range(1, epochs + 1):
                model.train()
                for xb, yb_s in loader_tr:
                    xb = xb.to(device).float()
                    yb_s = yb_s.to(device).float()
                    opt.zero_grad()
                    preds_s = model(xb)
                    loss = loss_fn(preds_s, yb_s)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                scheduler.step()

                # Calculate val_RMSE on original scale
                model.eval()
                val_preds, val_truth = [], []
                with torch.no_grad():
                    for xb, yb_orig in loader_val:
                        xb = xb.to(device).float()
                        pred_s = model(xb).cpu().numpy()
                        pred_orig = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
                        val_preds.append(pred_orig)
                        val_truth.append(yb_orig.numpy())
                val_preds = np.concatenate(val_preds) if val_preds else np.array([])
                val_truth = np.concatenate(val_truth) if val_truth else np.array([])
                val_rmse = root_mean_squared_error(val_truth, val_preds) if len(val_preds) else np.inf

                stopper.step(val_rmse, model)
                if stopper.should_stop():
                    print(f"[NN-REG] Early stopped at epoch {ep}, best val_RMSE={stopper.best_metric:.4f}")
                    break

            stopper.load_best(model, device)

            model.eval()
            with torch.no_grad():
                xt = torch.from_numpy(X_te_s).to(device).float()
                pred_s = model(xt).cpu().numpy()
                pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
            return pred

        cnn = CNN1DRegressor(seq_len=self.X_tr.shape[1])
        pred_y = _fit_reg(cnn, epochs=nn_epochs)
        results["CNN1D"] = {
            "r2":   r2_score(self.y_test, pred_y),
            "rmse": root_mean_squared_error(self.y_test, pred_y)
        }

        tfm = TransformerRegressor(seq_len=self.X_tr.shape[1])
        pred_y = _fit_reg(tfm, epochs=nn_epochs)
        results["Transformer"] = {
            "r2":   r2_score(self.y_test, pred_y),
            "rmse": root_mean_squared_error(self.y_test, pred_y)
        }

        for k, v in results.items():
            if isinstance(v, dict):
                print(f"{k}: R² = {v['r2']:.4f}, RMSE = {v['rmse']:.4f}")
            else:
                print(f"{k}: Accuracy = {v:.2%}")
        return results

# =============================================================================
# 3) Anomaly Detection (ML=GridSearch on train; NN uses val early stop/threshold)
# =============================================================================
class Autoencoder(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_feats, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, n_feats)
        )
    def forward(self, x):
        return self.dec(self.enc(x))

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.enc = nn.LSTM(input_size=n_feats, hidden_size=32, batch_first=True)
        self.dec = nn.LSTM(input_size=32, hidden_size=n_feats, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, n_feats)
        out, _ = self.enc(x)
        out, _ = self.dec(out)
        return out.squeeze(1)

class AnomalyModelPipeline:
    def __init__(self, train_data, test_data, val_data=None,
                 val_size=0.2, val_random_state=42, early_stop_patience=5, cv_folds=5):
        X_tr_full = np.array([d["x"] for d in train_data], dtype=np.float32)
        y_tr_full = np.array([1 if str(d.get("label", 0)).lower() in ("true","1","t","yes") else 0 for d in train_data], dtype=int)
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([1 if str(d.get("label", 0)).lower() in ("true","1","t","yes") else 0 for d in test_data], dtype=int)

        if val_data is None:
            self.X_tr, self.X_val, self.y_tr, self.y_val = train_test_split(
                X_tr_full, y_tr_full, test_size=val_size,
                random_state=val_random_state, stratify=y_tr_full
            )
        else:
            self.X_tr, self.y_tr = X_tr_full, y_tr_full
            self.X_val = np.array([d["x"] for d in val_data], dtype=np.float32)
            self.y_val = np.array([1 if str(d.get("label", 0)).lower() in ("true","1","t","yes") else 0 for d in val_data], dtype=int)

        self.early_stop_patience = early_stop_patience
        self.cv_folds = cv_folds

    def _grid_search_anomaly_ml(self, X_train_flat, y_train):
        """Perform supervised grid search for anomaly detection models on training set (using ROC-AUC as scoring)."""
        results = {}
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        # IsolationForest
        # Use GridSearchCV + scoring='roc_auc' with y_train (supervised parameter selection)
        iso = IsolationForest(random_state=42)
        param_iso = {
            "n_estimators": [200, 500, 800],
            "max_samples": ["auto", 0.6, 0.8, 1.0],
            "contamination": [0.01, 0.03, 0.05, 0.1]
        }
        gs_iso = GridSearchCV(iso, param_grid=param_iso, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs_iso.fit(X_train_flat, y_train)
        results["IsolationForest"] = gs_iso

        # OneClassSVM (with standardization)
        pipe_oc = Pipeline([("scaler", StandardScaler()), ("ocsvm", OneClassSVM())])
        param_oc = {
            "ocsvm__kernel": ["rbf"],
            "ocsvm__gamma": ["scale", "auto", 0.01, 0.1, 1],
            "ocsvm__nu": [0.01, 0.05, 0.1, 0.2]
        }
        gs_oc = GridSearchCV(pipe_oc, param_grid=param_oc, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        gs_oc.fit(X_train_flat, y_train)
        results["OneClassSVM"] = gs_oc

        return results

    def train_and_evaluate(self, epochs_ae=100, batch_size=32):
        results = {}

        # ---------------- Traditional anomaly detection: GridSearchCV on train only, evaluate on test ----------------
        flat_tr  = self.X_tr.reshape(len(self.X_tr), -1)
        flat_te  = self.X_test.reshape(len(self.X_test), -1)

        best_ml = self._grid_search_anomaly_ml(flat_tr, self.y_tr)
        for name, gs in best_ml.items():
            # Convert to {0,1} predictions: if decision_function / predict can be used directly
            if hasattr(gs.best_estimator_, "predict"):
                raw = gs.predict(flat_te)
                # For iso/ocsvm: predict returns {1, -1}, convert to {0,1}
                if set(np.unique(raw)) == {-1, 1}:
                    pred_bin = (raw == -1).astype(int)
                    # AUC needs continuous scores, if decision_function is available, recalculate
                    if hasattr(gs.best_estimator_, "decision_function"):
                        scores = gs.best_estimator_.decision_function(flat_te)
                        auc = roc_auc_score(self.y_test, scores)
                    else:
                        auc = roc_auc_score(self.y_test, pred_bin)
                    prec = precision_score(self.y_test, pred_bin)
                else:
                    # Already probability/score case (rare)
                    scores = raw
                    thr = np.percentile(scores, 95)
                    pred_bin = (scores > thr).astype(int)
                    auc = roc_auc_score(self.y_test, scores)
                    prec = precision_score(self.y_test, pred_bin)
            else:
                # fallback: use decision_function
                scores = gs.best_estimator_.decision_function(flat_te)
                thr = np.percentile(scores, 95)
                pred_bin = (scores > thr).astype(int)
                auc = roc_auc_score(self.y_test, scores)
                prec = precision_score(self.y_test, pred_bin)

            results[name] = {"precision": prec, "auc": auc}
            print(f"[ML][{name}] best_params={gs.best_params_}, CV_best_AUC={gs.best_score_:.4f}, Test_Prec={prec:.4f}, Test_AUC={auc:.4f}")

        # ---------------- NN anomaly detection: train on train, early stop on val, threshold on test ----------------
        x_scaler = StandardScaler()
        tr_s  = x_scaler.fit_transform(self.X_tr)       # Fit only on train
        val_s = x_scaler.transform(self.X_val)
        te_s  = x_scaler.transform(self.X_test)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ds_tr  = TensorDataset(torch.from_numpy(tr_s))
        ds_val = TensorDataset(torch.from_numpy(val_s))
        loader_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=False)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)

        def _auc_by_recon_err(model, X_val_np, y_val_np) -> float:
            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(X_val_np).to(device).float()
                recon = model(x).cpu().numpy()
            err = ((recon - X_val_np) ** 2).mean(axis=1)
            return roc_auc_score(y_val_np, err)

        def _fit_autoencoder(model, epochs=epochs_ae):
            model = model.to(device)
            opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            loss_fn = nn.MSELoss()
            stopper = EarlyStopper(mode="max", patience=self.early_stop_patience)

            for ep in range(1, epochs + 1):
                model.train()
                for (xb,) in loader_tr:
                    xb = xb.to(device).float()
                    opt.zero_grad()
                    recon = model(xb)
                    loss = loss_fn(recon, xb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                val_auc = _auc_by_recon_err(model, val_s, self.y_val)
                stopper.step(val_auc, model)
                if stopper.should_stop():
                    print(f"[NN-ANO] Early stopped at epoch {ep}, best val_AUC={stopper.best_metric:.4f}")
                    break

            stopper.load_best(model, device)
            return model

        # AE
        ae = Autoencoder(n_feats=self.X_tr.shape[1])
        ae = _fit_autoencoder(ae, epochs=epochs_ae)
        with torch.no_grad():
            xt = torch.from_numpy(te_s).to(device).float()
            recon = ae(xt).cpu().numpy()
        err = ((recon - te_s) ** 2).mean(axis=1)
        with torch.no_grad():
            xv = torch.from_numpy(val_s).to(device).float()
            recon_v = ae(xv).cpu().numpy()
        err_v = ((recon_v - val_s) ** 2).mean(axis=1)
        thr = np.percentile(err_v, 95)
        p_ae = (err > thr).astype(int)
        results["Autoencoder"] = {
            "precision": precision_score(self.y_test, p_ae),
            "auc":       roc_auc_score(self.y_test, err)
        }

        # LSTM-AE
        lae = LSTMAutoencoder(n_feats=self.X_tr.shape[1])
        lae = _fit_autoencoder(lae, epochs=epochs_ae)
        with torch.no_grad():
            xt = torch.from_numpy(te_s).to(device).float()
            recon = lae(xt).cpu().numpy()
        err = ((recon - te_s) ** 2).mean(axis=1)
        with torch.no_grad():
            xv = torch.from_numpy(val_s).to(device).float()
            recon_v = lae(xv).cpu().numpy()
        err_v = ((recon_v - val_s) ** 2).mean(axis=1)
        thr = np.percentile(err_v, 95)
        p_lae = (err > thr).astype(int)
        results["LSTM_Autoencoder"] = {
            "precision": precision_score(self.y_test, p_lae),
            "auc":       roc_auc_score(self.y_test, err)
        }

        for name, m in results.items():
            print(f"{name}: Precision = {m['precision']:.4f}, AUC = {m['auc']:.4f}")
        return results