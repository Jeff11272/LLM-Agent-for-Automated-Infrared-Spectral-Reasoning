#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cls_dataset.py

Unified dataset interface: input labels, total_samples, split_ratio are provided externally.
"""

import numpy as np
from typing import List, Any, Tuple,Dict,Optional

class CLS_Dataset:
    def __init__(
        self,
        feature: np.ndarray,
        labels: list,
        total_samples: int,
        split_ratio: list,
        random_seed: Optional[int] = 3,
    ):
        if random_seed is None:
            random_seed = 3
        """
        Parameters:
        - feature: numpy array of shape (n_classes, n_samples_per_class, n_features)
        - labels: list of class labels
        - total_samples: total number of samples for all classes
        - split_ratio: [train_ratio, val_ratio, test_ratio], sum to 1
        - random_seed: int for reproducibility
        """
        self.labels = labels
        self.n_classes = len(labels)
        self.random_seed = random_seed
        # Randomly assign sample count for each class
        base_num = total_samples // self.n_classes
        remain = total_samples % self.n_classes
        np.random.seed(self.random_seed)
        class_counts = [base_num] * self.n_classes
        if remain > 0:
            extra_idx = np.random.choice(self.n_classes, remain, replace=False)
            for idx in extra_idx:
                class_counts[idx] += 1
        self.class_counts = class_counts  # samples per class

        # feature shape: (n_classes, n_samples_per_class, n_features)
        self.X_pca = feature

        # Calculate split counts for all samples
        ratios = np.array(split_ratio)
        assert np.isclose(ratios.sum(), 1.0), "split_ratio must sum to 1"
        self.n_train = int(total_samples * ratios[0])
        self.n_val   = int(total_samples * ratios[1])
        self.n_test  = total_samples - self.n_train - self.n_val

        self.dataset = []
        self.true_labels_val = []
        self.true_labels_test = []

        self._build_dataset()
        self._split_data()

    def _build_dataset(self):
        np.random.seed(self.random_seed)
        all_items = []
        # Collect all samples with label info
        for cls_id, count in enumerate(self.class_counts):
            label = self.labels[cls_id]
            available_idx = np.arange(self.X_pca.shape[1])
            chosen_idx = np.random.choice(available_idx, count, replace=False)
            for i, feat_idx in enumerate(chosen_idx):
                item = {
                    "name": f"{label}_{i}",
                    "x": self.X_pca[cls_id][feat_idx].tolist(),
                    "label": label,
                }
                all_items.append(item)
        # Shuffle all samples
        np.random.shuffle(all_items)

        # Split by class to ensure every split contains every class
        train_data, val_data, test_data = [], [], []
        class_items = {label: [] for label in self.labels}
        for item in all_items:
            class_items[item["label"]].append(item)

        # Calculate per-class split counts
        train_per_class = self.n_train // self.n_classes
        val_per_class = self.n_val // self.n_classes
        test_per_class = self.n_test // self.n_classes

        # If not divisible, randomly assign the remainder
        remain_train = self.n_train - train_per_class * self.n_classes
        remain_val = self.n_val - val_per_class * self.n_classes
        remain_test = self.n_test - test_per_class * self.n_classes

        np.random.seed(self.random_seed)
        train_extra = np.random.choice(self.labels, remain_train, replace=False) if remain_train > 0 else []
        val_extra = np.random.choice(self.labels, remain_val, replace=False) if remain_val > 0 else []
        test_extra = np.random.choice(self.labels, remain_test, replace=False) if remain_test > 0 else []

        for label in self.labels:
            items = class_items[label]
            np.random.shuffle(items)
            train_count = train_per_class + (1 if label in train_extra else 0)
            val_count = val_per_class + (1 if label in val_extra else 0)
            test_count = test_per_class + (1 if label in test_extra else 0)
            train_data.extend([dict(**item, split="train") for item in items[:train_count]])
            val_data.extend([dict(**item, split="val") for item in items[train_count:train_count+val_count]])
            test_data.extend([dict(**item, split="test") for item in items[train_count+val_count:]])

        # Combine all splits
        self.dataset = train_data + val_data + test_data

        # Store true labels for val/test
        self.true_labels_val = [item["label"] for item in val_data]
        self.true_labels_test = [item["label"] for item in test_data]

    def _split_data(self):
        # Separate by split
        self.train_data = [d for d in self.dataset if d["split"] == "train"]
        self.val_data = [d for d in self.dataset if d["split"] == "val"]
        self.test_data = [d for d in self.dataset if d["split"] == "test"]

        # Shuffle val and test
        np.random.seed(self.random_seed)
        perm_val = np.random.permutation(len(self.val_data))
        perm_test = np.random.permutation(len(self.test_data))

        self.val_data = [self.val_data[i] for i in perm_val]
        self.true_labels_val = [self.true_labels_val[i] for i in perm_val]

        self.test_data = [self.test_data[i] for i in perm_test]
        self.true_labels_test = [self.true_labels_test[i] for i in perm_test]

    def summary(self):
        """Print summary of dataset sizes and labels."""
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}, labels: {self.true_labels_val}")
        print(f"Test samples: {len(self.test_data)}, labels: {self.true_labels_test}")

class REG_Dataset:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        total_samples: int,
        split_ratio: list,
        random_seed: Optional[int] = 5,
    ):
        if random_seed is None:
            random_seed = 5
        """
        Parameters:
        - X: numpy array of shape (1, total_samples, n_features) or (total_samples, n_features)
        - Y: numpy array of shape (1, total_samples, 1) or (total_samples,) or (total_samples, 1)
        - total_samples: number of samples
        - split_ratio: [train_ratio, val_ratio, test_ratio], sum to 1
        - random_seed: for reproducible shuffling
        """
        # Process X: remove leading class dimension if present
        if X.ndim == 3:
            assert X.shape[0] == 1, "X first dimension must be 1 for single-class data"
            X_proc = X[0]
        else:
            X_proc = X
        assert X_proc.ndim == 2, f"X must be 2D after squeezing, but got shape {X_proc.shape}"

        # Process Y
        if Y.ndim == 3:
            assert Y.shape[0] == 1 and Y.shape[2] == 1, \
                "Y must have shape (1, samples, 1) for single-class data"
            Y_flat = Y[0, :, 0]
        else:
            Y_flat = Y.squeeze()
        assert Y_flat.ndim == 1, f"Y must be 1D after squeezing, but got shape {Y_flat.shape}"

        self.X = X_proc
        self.Y = Y_flat
        self.total_samples, self.n_features = self.X.shape

        # Calculate split counts
        ratios = np.array(split_ratio)
        assert np.isclose(ratios.sum(), 1.0), "split_ratio must sum to 1"
        self.n_train = int(total_samples * ratios[0])
        self.n_val   = int(total_samples * ratios[1])
        self.n_test  = total_samples - self.n_train - self.n_val

        assert self.n_train + self.n_val + self.n_test <= self.total_samples, (
            f"Not enough samples: needed {self.n_train + self.n_val + self.n_test} but only {self.total_samples} available"
        )

        self.random_seed = random_seed

        # Shuffle and split indices
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.total_samples)
        train_idx = indices[:self.n_train]
        val_idx   = indices[self.n_train:self.n_train + self.n_val]
        test_idx  = indices[self.n_train + self.n_val:self.n_train + self.n_val + self.n_test]

        # Build data items
        self.train_data = self._build_items(train_idx, "train")
        self.val_data   = self._build_items(val_idx,   "val")
        self.test_data  = self._build_items(test_idx,  "test")

        # Store true labels
        self.y_val_true  = [item['y'] for item in self.val_data]
        self.y_test_true = [item['y'] for item in self.test_data]

    def _build_items(self, idx_list, split_name):
        items = []
        for idx in idx_list:
            name = f"sample{idx}_{split_name}"
            x = self.X[idx].tolist()
            y = float(self.Y[idx])
            items.append({
                'name': name,
                'x': x,
                'y': y,
                'split': split_name
            })
        return items

    def summary(self):
        """Print a summary of dataset sizes and true labels."""
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}, true y: {self.y_val_true}")
        print(f"Test samples: {len(self.test_data)}, true y: {self.y_test_true}")



class ANO_Dataset:
    def __init__(
        self,
        X: np.ndarray,
        labels: List[Any],
        total_samples: int,
        split_ratio: List[float],
        normal_class: int = 0,
        noise_std: float = 0.05,
        random_seed: Optional[int] = 42,
    ):
        if random_seed is None:
            random_seed = 42
        assert X.ndim == 3, "X must be a 3D array (n_classes, n_samples, n_features)"
        assert len(split_ratio) == 3 and abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1"

        self.X = X
        self.labels = labels
        self.n_classes = len(labels)
        self.normal_class = normal_class
        self.noise_std = noise_std
        self.random_seed = random_seed

        rng = np.random.RandomState(self.random_seed)

        # 1) 从 normal 类抽取总样本
        n_total = min(total_samples, X.shape[1])
        idxs_normal = np.arange(X.shape[1])
        chosen_normal = rng.choice(idxs_normal, n_total, replace=False)

        n_norm  = int(n_total * 0.75)
        n_intra = int(n_total * 0.25)
        n_inter = n_intra*2

        norm_idxs  = chosen_normal[:n_norm]
        intra_idxs = chosen_normal[n_norm:n_norm + n_intra]

        # inter 候选池
        inter_pool = []
        for cls in range(self.n_classes):
            if cls == self.normal_class:
                continue
            for i in range(X.shape[1]):
                inter_pool.append((cls, i))
        inter_pool = np.array(inter_pool, dtype=object)
        if n_inter > 0:
            chosen_inter = rng.choice(len(inter_pool), n_inter, replace=False)
            inter_idxs = inter_pool[chosen_inter]
        else:
            inter_idxs = np.array([], dtype=object)

        # 3) 构造条目
        normal_entries = []
        for i in norm_idxs:
            x_vec = X[self.normal_class, i, :]
            normal_entries.append({
                "name": f"class{self.normal_class}_idx{i}_normal",
                "x": x_vec.tolist(),
                "label": "false",
                "class_id": int(self.normal_class),
                "sample_idx": int(i),
                "anomaly_type": "normal"
            })

        intra_entries = []
        for i in intra_idxs:
            x_vec = X[self.normal_class, i, :] + rng.normal(0, self.noise_std, size=X.shape[2])
            intra_entries.append({
                "name": f"class{self.normal_class}_idx{i}_intra",
                "x": x_vec.tolist(),
                "label": "true",
                "class_id": int(self.normal_class),
                "sample_idx": int(i),
                "anomaly_type": "intra"
            })

        inter_entries = []
        for pair in inter_idxs:
            cls = int(pair[0]); i = int(pair[1])
            x_vec = X[cls, i, :]
            inter_entries.append({
                "name": f"class{cls}_idx{i}_inter",
                "x": x_vec.tolist(),
                "label": "true",
                "class_id": int(cls),
                "sample_idx": int(i),
                "anomaly_type": "inter"
            })

        # 4) 桶内打乱（可复现）
        rng.shuffle(normal_entries)
        rng.shuffle(intra_entries)
        rng.shuffle(inter_entries)

        # === 方案B：最大余数法分配配额（全局精确命中） ===
        def lrm_allocate(
            counts: Dict[str, int],
            ratio: float,
            global_target: int,
            capacity: Dict[str, int] = None
        ) -> Dict[str, int]:
            """
            Largest Remainder Method with per-bucket capacity.
            counts: 每个桶的总样本数
            ratio: 目标比例（如 0.6）
            global_target: 全局想要的个数（如 floor(n_total*0.6)）
            capacity: 每个桶的可用容量（不传则等于 counts）
            """
            keys = list(counts.keys())
            cap = capacity if capacity is not None else counts

            shares = {k: counts[k] * ratio for k in keys}
            base_unclamped = {k: int(np.floor(shares[k])) for k in keys}
            alloc = {k: min(base_unclamped[k], cap[k]) for k in keys}

            # 余数（若没容量则设为 -1，防止再分）
            remainders = {
                k: (shares[k] - base_unclamped[k]) if (cap[k] - alloc[k]) > 0 else -1.0
                for k in keys
            }

            # 当前总量 & 需要补的名额
            cur = sum(alloc.values())
            need = min(global_target, sum(cap.values()))
            extras = need - cur

            # 依余数从大到小、逐个补齐
            while extras > 0:
                # 找到有剩余容量的最大余数桶
                k_star = None
                best_r = -1.0
                for k in keys:
                    if (cap[k] - alloc[k]) > 0 and remainders[k] > best_r:
                        best_r = remainders[k]
                        k_star = k
                if k_star is None:
                    break  # 没容量了（理论上不会发生）
                alloc[k_star] += 1
                extras -= 1
                # 若该桶已满，余数置为 -1
                if cap[k_star] - alloc[k_star] == 0:
                    remainders[k_star] = -1.0
            return alloc

        # 各桶样本数
        counts = {
            "normal": len(normal_entries),
            "intra":  len(intra_entries),
            "inter":  len(inter_entries),
        }

        # 全局目标配额
        train_ratio, val_ratio, test_ratio = split_ratio
        g_train = int(n_total * train_ratio)
        g_val   = int(n_total * val_ratio)
        g_test  = n_total - g_train - g_val  # 保证三者求和为 n_total

        # 先分配 Train
        train_alloc = lrm_allocate(counts, train_ratio, g_train, capacity=counts)

        # 再分配 Val（容量 = counts - train_alloc）
        cap_val = {k: counts[k] - train_alloc[k] for k in counts}
        val_alloc = lrm_allocate(counts, val_ratio, g_val, capacity=cap_val)

        # Test 用剩余容量补齐
        test_alloc = {k: counts[k] - train_alloc[k] - val_alloc[k] for k in counts}

        # 5) 按分配数量切片
        def cut(entries: List[dict], n_train: int, n_val: int):
            train = entries[:n_train]
            val = entries[n_train:n_train + n_val]
            test = entries[n_train + n_val:]
            return train, val, test

        t_norm, v_norm, s_norm = cut(normal_entries, train_alloc["normal"], val_alloc["normal"])
        t_intra, v_intra, s_intra = cut(intra_entries, train_alloc["intra"], val_alloc["intra"])
        t_inter, v_inter, s_inter = cut(inter_entries, train_alloc["inter"], val_alloc["inter"])

        # 6) 合并 + 全局打乱
        self.train_data = t_norm + t_intra + t_inter
        self.val_data   = v_norm + v_intra + v_inter
        self.test_data  = s_norm + s_intra + s_inter

        rng.shuffle(self.train_data)
        rng.shuffle(self.val_data)
        rng.shuffle(self.test_data)

        self.train_data=self.train_data
        
        # 7) 评估用布尔标签
        def labels_to_bool(list_of_entries: List[dict]) -> List[bool]:
            out = []
            for d in list_of_entries:
                lab = d.get("label")
                if isinstance(lab, str):
                    out.append(lab.strip().lower() in ("true", "1", "t", "yes"))
                elif isinstance(lab, (int, float)):
                    out.append(int(lab) == 1)
                elif isinstance(lab, bool):
                    out.append(lab)
                else:
                    out.append(False)
            return out

        self.y_val = labels_to_bool(self.val_data)
        self.y_test = labels_to_bool(self.test_data)

    def summary(self):
        def count_types(data):
            n_norm = sum(d["anomaly_type"] == "normal" for d in data)
            n_intra = sum(d["anomaly_type"] == "intra" for d in data)
            n_inter = sum(d["anomaly_type"] == "inter" for d in data)
            return n_norm, n_intra, n_inter
        tn, ti, te = count_types(self.train_data)
        vn, vi, ve = count_types(self.val_data)
        sn, si, se = count_types(self.test_data)
        print(f"Train: {len(self.train_data)} ({tn} normal, {ti} intra, {te} inter)")
        print(f"Val:   {len(self.val_data)} ({vn} normal, {vi} intra, {ve} inter)")
        print(f"Test:  {len(self.test_data)} ({sn} normal, {si} inter, {se} inter)")

