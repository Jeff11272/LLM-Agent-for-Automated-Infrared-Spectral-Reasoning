import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# 1. Classification
# -----------------------------------------------------------------------------
class CNN1DClassifier(nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
            nn.Flatten(), nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1))


class TransformerClassifier(nn.Module):
    def __init__(self, n_feats, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, 128)
        encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.enc = nn.TransformerEncoder(encoder, num_layers=2)
        self.fc2 = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.fc1(x).unsqueeze(1)
        x = self.enc(x).squeeze(1)
        return self.fc2(x)


class ClassificationModelPipeline:
    def __init__(self, train_data, test_data):
        self.X_train = np.array([d["x"] for d in train_data], dtype=np.float32)
        self.y_train = np.array([d["label"] for d in train_data])
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([d["label"] for d in test_data])

        labels = sorted(set(self.y_train))
        self.l2i = {l:i for i,l in enumerate(labels)}
        self.y_train_i = np.array([self.l2i[y] for y in self.y_train], dtype=np.int64)
        self.y_test_i  = np.array([self.l2i[y] for y in self.y_test],  dtype=np.int64)

        self.models = {
            "SVM":            SVC(kernel="rbf"),
            "KNN":            KNeighborsClassifier(),
            "RandomForest":   RandomForestClassifier(n_estimators=100, random_state=42)
        }

    def train_and_evaluate(self):
        results = {}
        flat_train = self.X_train.reshape(len(self.X_train), -1)
        flat_test  = self.X_test.reshape(len(self.X_test), -1)

        for name, mdl in self.models.items():
            mdl.fit(flat_train, self.y_train_i)
            pred = mdl.predict(flat_test)
            results[name] = accuracy_score(self.y_test_i, pred)

        # ---- PyTorch part ----
        # ensure labels are long dtype
        ds_train = TensorDataset(torch.from_numpy(self.X_train),
                                 torch.from_numpy(self.y_train_i).long())
        loader = DataLoader(ds_train, batch_size=32, shuffle=True)
        X_test_t = torch.from_numpy(self.X_test)

        # CNN1D (3-layer)
        cnn = CNN1DClassifier(self.X_train.shape[1], len(self.l2i))
        opt = optim.Adam(cnn.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        cnn.train()
        for _ in range(20):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(cnn(xb), yb).backward()
                opt.step()
        cnn.eval()
        with torch.no_grad():
            pred = cnn(X_test_t).argmax(dim=1).numpy()
        results["CNN1D"] = accuracy_score(self.y_test_i, pred)

        # Transformer‑MLP
        tfm = TransformerClassifier(self.X_train.shape[1], len(self.l2i))
        opt = optim.Adam(tfm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        tfm.train()
        for _ in range(20):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(tfm(xb), yb).backward()
                opt.step()
        tfm.eval()
        with torch.no_grad():
            pred = tfm(X_test_t).argmax(dim=1).numpy()
        results["Transformer"] = accuracy_score(self.y_test_i, pred)

        for k, v in results.items():
            print(f"{k}: Accuracy = {v:.2%}")
        return results


# -----------------------------------------------------------------------------
# 2. Regression
# -----------------------------------------------------------------------------
class CNN1DRegressor(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.fc1 = nn.Linear(n_feats, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.fc1(x).unsqueeze(1)
        x = self.enc(x).squeeze(1)
        return self.fc2(x).squeeze(1)


class RegressionModelPipeline:
    def __init__(self, train_data, test_data):
        self.X_train = np.array([d["x"] for d in train_data], dtype=np.float32)
        self.y_train = np.array([d["y"] for d in train_data], dtype=np.float32)
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([d["y"] for d in test_data], dtype=np.float32)

        self.models = {
            "LinearRegression":      LinearRegression(),
            "SVR":                   SVR(),
            "PLSR":                  PLSRegression(n_components=2),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42)
        }

    def train_and_evaluate(self):
        results = {}
        flat_train = self.X_train.reshape(len(self.X_train), -1)
        flat_test  = self.X_test.reshape(len(self.X_test), -1)

        for name, mdl in self.models.items():
            mdl.fit(flat_train, self.y_train)
            pred = mdl.predict(flat_test)
            results[name] = {
                "r2":   r2_score(self.y_test, pred),
                "rmse": mean_squared_error(self.y_test, pred, squared=False)
            }

        ds_train = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        loader = DataLoader(ds_train, batch_size=32, shuffle=True)

        # CNN1D
        cnn = CNN1DRegressor(self.X_train.shape[1])
        opt = optim.Adam(cnn.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        cnn.train()
        for _ in range(20):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(cnn(xb), yb).backward()
                opt.step()
        cnn.eval()
        with torch.no_grad():
            pred = cnn(torch.from_numpy(self.X_test)).numpy()
        results["CNN1D"] = {
            "r2":   r2_score(self.y_test, pred),
            "rmse": mean_squared_error(self.y_test, pred, squared=False)
        }

        # Transformer-MLP
        tfm = TransformerRegressor(self.X_train.shape[1])
        opt = optim.Adam(tfm.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        tfm.train()
        for _ in range(20):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(tfm(xb), yb).backward()
                opt.step()
        tfm.eval()
        with torch.no_grad():
            pred = tfm(torch.from_numpy(self.X_test)).numpy()
        results["Transformer"] = {
            "r2":   r2_score(self.y_test, pred),
            "rmse": mean_squared_error(self.y_test, pred, squared=False)
        }

        for k, v in results.items():
            print(f"{k}: R² = {v['r2']:.4f}, RMSE = {v['rmse']:.4f}")
        return results


# -----------------------------------------------------------------------------
# 3. Anomaly Detection
# -----------------------------------------------------------------------------
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
        # x: (batch, n_feats)
        x = x.unsqueeze(1)  # (batch, 1, n_feats)
        out, _ = self.enc(x)  # (batch, 1, 32)
        out, _ = self.dec(out)  # (batch, 1, n_feats)
        return out.squeeze(1)  # (batch, n_feats)



class AnomalyModelPipeline:
    def __init__(self, train_data, test_data):
        self.X_train = np.array([d["x"] for d in train_data], dtype=np.float32)
        self.y_train = np.array([d["label"] for d in train_data], dtype=int)
        self.X_test  = np.array([d["x"] for d in test_data], dtype=np.float32)
        self.y_test  = np.array([d["label"] for d in test_data], dtype=int)

    def train_and_evaluate(self):
        results = {}

        flat_train = self.X_train.reshape(len(self.X_train), -1)
        flat_test  = self.X_test.reshape(len(self.X_test), -1)

        iso = IsolationForest(random_state=42).fit(flat_train)
        p_iso = iso.predict(flat_test) == -1
        results["IsolationForest"] = {
            "precision": precision_score(self.y_test, p_iso),
            "auc":       roc_auc_score(self.y_test, p_iso)
        }

        oc = OneClassSVM().fit(flat_train)
        p_oc = oc.predict(flat_test) == -1
        results["OneClassSVM"] = {
            "precision": precision_score(self.y_test, p_oc),
            "auc":       roc_auc_score(self.y_test, p_oc)
        }

        # Autoencoder
        ds_train = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.X_train))
        loader = DataLoader(ds_train, batch_size=32, shuffle=True)
        ae = Autoencoder(self.X_train.shape[1])
        opt = optim.Adam(ae.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        ae.train()
        for _ in range(20):
            for xb, _ in loader:
                opt.zero_grad()
                loss_fn(ae(xb), xb).backward()
                opt.step()
        ae.eval()
        with torch.no_grad():
            recon = ae(torch.from_numpy(self.X_test))
        err = ((recon.numpy() - self.X_test)**2).mean(axis=1)
        p_ae = err > np.percentile(err, 95)
        results["Autoencoder"] = {
            "precision": precision_score(self.y_test, p_ae),
            "auc":       roc_auc_score(self.y_test, p_ae)
        }

        # LSTM Autoencoder
        lae = LSTMAutoencoder(self.X_train.shape[1])
        opt = optim.Adam(lae.parameters(), lr=1e-3)
        lae.train()
        for _ in range(20):
            for xb, _ in loader:
                opt.zero_grad()
                loss_fn(lae(xb), xb).backward()
                opt.step()
        lae.eval()
        with torch.no_grad():
            recon = lae(torch.from_numpy(self.X_test))
        err = ((recon.numpy() - self.X_test)**2).mean(axis=1)
        p_lae = err > np.percentile(err, 95)
        results["LSTM_Autoencoder"] = {
            "precision": precision_score(self.y_test, p_lae),
            "auc":       roc_auc_score(self.y_test, p_lae)
        }

        for name, m in results.items():
            print(f"{name}: Precision = {m['precision']:.4f}, AUC = {m['auc']:.4f}")
        return results
