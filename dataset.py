
import numpy as np

class CLS_Dataset:
    def __init__(
        self,
        feature: np.ndarray,
        label_type: str,
        n_classes: int = 3,
        n_train_per_class: int = 3,
        n_val_per_class: int = 2,
        n_test_per_class: int = 2,
        random_seed: int = 4
    ):
        """
        Initialize dataset splitting.

        Parameters:
        - X_pca: numpy array of shape (n_classes, total_per_class, n_features)
        - label_type: one of ['Ink', 'Chenpi', 'CN_medicine', 'Puer_tea']
        - n_classes, n_train_per_class, n_val_per_class, n_test_per_class: split sizes
        - random_seed: int for reproducibility
        """
        self.X_pca = feature
        self.label_type = label_type
        self.n_classes = n_classes
        self.n_train = n_train_per_class
        self.n_val = n_val_per_class
        self.n_test = n_test_per_class
        self.total_per_class = n_train_per_class + n_val_per_class + n_test_per_class
        self.random_seed = random_seed

        # Prepare label lists based on label_type
        self.labels = self._generate_labels(label_type)

        # Build dataset
        self.dataset = []
        self.true_labels_val = []
        self.true_labels_test = []

        self._build_dataset()
        self._split_data()

    def _generate_labels(self, label_type):
        if label_type == 'Ink':
            return ["red1", "red2", "red3",
                    "red4", "red5", "red6",
                    "red7", "red8", "red9",
                    "red10", "red11", "red12",]
        elif label_type == 'Chenpi':
            return ["1", "2", "3",
                    "4", "5", "6",
                    "7", "8",]
        elif label_type == 'CN_medicine':
            return ["山银花", "金银花", "山银花金银花混合物"]
        elif label_type == 'Puer_tea':
            return ["1", "2", "3",'4','5', '6','7', '8', '9','10','11','12','13','14']
        else:
            raise ValueError(f"Unknown label_type: {label_type}")

    def _build_dataset(self):
        for cls_id in range(self.n_classes):
            label = self.labels[cls_id]
            for i in range(self.total_per_class):
                idx = cls_id * self.total_per_class + i
                if i < self.n_train:
                    split = "train"
                elif i < self.n_train + self.n_val:
                    split = "val"
                else:
                    split = "test"
                name = f"{label}_{split}_{i}"
                item = {
                    "name": name,
                    "x": self.X_pca[cls_id][i].tolist(),
                    "label": label,
                    "split": split
                }
                self.dataset.append(item)
                if split == "val":
                    self.true_labels_val.append(label)
                elif split == "test":
                    self.true_labels_test.append(label)

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
        n_train: int = 20,
        n_val: int = 5,
        n_test: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize regression dataset splitting for a single material class.

        Parameters:
        - X: numpy array of shape (1, total_samples, n_features) or (total_samples, n_features)
        - Y: numpy array of shape (1, total_samples, 1) or (total_samples,) or (total_samples, 1)
        - n_train: number of training samples
        - n_val: number of validation samples
        - n_test: number of test samples
        - random_seed: for reproducible shuffling
        """
        # Process X: remove leading class dimension if present
        if X.ndim == 3:
            assert X.shape[0] == 1, "X first dimension must be 1 for single-class data"
            X_proc = X[0]  # shape: (total_samples, n_features)
        else:
            X_proc = X
        assert X_proc.ndim == 2, f"X must be 2D after squeezing, but got shape {X_proc.shape}"

        # Process Y: expecting shape (1, total_samples, 1)
        if Y.ndim == 3:
            assert Y.shape[0] == 1 and Y.shape[2] == 1, \
                "Y must have shape (1, samples, 1) for single-class data"
            Y_flat = Y[0, :, 0]
        else:
            # Remove singleton dimension in second axis if present
            Y_flat = Y.squeeze()
        assert Y_flat.ndim == 1, f"Y must be 1D after squeezing, but got shape {Y_flat.shape}"

        self.X = X_proc
        self.Y = Y_flat
        self.total_samples, self.n_features = self.X.shape

        # Ensure enough samples
        needed = n_train + n_val + n_test
        assert needed <= self.total_samples, (
            f"Not enough samples: needed {needed} but only {self.total_samples} available"
        )

        self.n_train = n_train
        self.n_val   = n_val
        self.n_test  = n_test
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
        normal_class: int = 0,
        n_train_norm: int = 20,
        n_val_norm:   int = 5,
        n_test_norm:  int = 5,
        n_inter_anom: int = 10,
        n_intra_anom: int = 5,
        noise_std:    float = 0.01,
        random_seed:  int = 42
    ):
        assert X.ndim == 3, "X must be 3D array"
        self.X = X
        self.n_classes, self.n_samples, self.n_features = X.shape
        assert 0 <= normal_class < self.n_classes, "normal_class out of range"

        # 参数
        self.normal_class  = normal_class
        self.n_train_norm  = n_train_norm
        self.n_val_norm    = n_val_norm
        self.n_test_norm   = n_test_norm
        self.n_inter_anom  = n_inter_anom
        self.n_intra_anom  = n_intra_anom
        self.noise_std     = noise_std
        self.random_seed   = random_seed

        np.random.seed(self.random_seed)
        self._sample_indices()
        self._assemble_splits()

    def _sample_indices(self):
        idxs = np.arange(self.n_samples)

        # 类内异常（抽取正常类样本）
        self.intra_idxs = list(
            np.random.choice(idxs, self.n_intra_anom, replace=False)
        )
        # 正常池
        self.normal_pool = list(np.setdiff1d(idxs, self.intra_idxs))
        needed_norm = self.n_train_norm + self.n_val_norm + self.n_test_norm
        assert needed_norm <= len(self.normal_pool), (
            f"Need {needed_norm} normals but only {len(self.normal_pool)} available"
        )
        sel_norm = list(np.random.choice(self.normal_pool, needed_norm, replace=False))
        perm = np.random.permutation(len(sel_norm))
        self.train_norm = [sel_norm[i] for i in perm[:self.n_train_norm]]
        self.val_norm   = [sel_norm[i] for i in perm[self.n_train_norm:
                                                     self.n_train_norm+self.n_val_norm]]
        self.test_norm  = [sel_norm[i] for i in perm[self.n_train_norm+self.n_val_norm:]]

        # 类间异常（来自其他类别）
        self.inter_idxs = []
        per_cls = int(np.ceil(self.n_inter_anom / (self.n_classes - 1)))
        total = 0
        for cls in range(self.n_classes):
            if cls == self.normal_class:
                continue
            avail = np.arange(self.n_samples)
            pick = min(per_cls, self.n_inter_anom - total)
            chosen = np.random.choice(avail, pick, replace=False)
            self.inter_idxs += [(cls, int(i)) for i in chosen]
            total = len(self.inter_idxs)
            if total >= self.n_inter_anom:
                break

    def _make_entries(self, idxs, anomaly=False):
        entries = []
        for item in idxs:
            if isinstance(item, tuple):
                cls, i = item
            else:
                cls, i = self.normal_class, item
            # 获取原始光谱向量
            x_vec = self.X[cls, i, :]
            # 对类内异常注入噪声
            if anomaly and cls == self.normal_class:
                x_vec = x_vec + np.random.normal(0, self.noise_std, size=x_vec.shape)
            entries.append({
                "name":       f"class{cls}_idx{i}_{'anom' if anomaly else 'norm'}",
                "x":          x_vec.tolist(),
                "label":      anomaly,
                "class_id":   cls,
                "sample_idx": i
            })
        return entries

    def _assemble_splits(self):
        # 训练集：仅正常样本
        self.train_data = self._make_entries(self.train_norm, anomaly=False)

        # 验证集：正常 + 半数异常
        half_intra = self.intra_idxs[:self.n_intra_anom // 2]
        half_inter = self.inter_idxs[:len(self.inter_idxs) // 2]
        self.val_data = (
            self._make_entries(self.val_norm, False)
            + self._make_entries(half_intra, True)
            + self._make_entries(half_inter, True)
        )
        self.y_val = [d["label"] for d in self.val_data]

        # 测试集：正常 + 剩余异常
        rem_intra = self.intra_idxs[self.n_intra_anom // 2:]
        rem_inter = self.inter_idxs[len(self.inter_idxs) // 2:]
        self.test_data = (
            self._make_entries(self.test_norm, False)
            + self._make_entries(rem_intra, True)
            + self._make_entries(rem_inter, True)
        )
        self.y_test = [d["label"] for d in self.test_data]

    def summary(self):
        print(f"Train (normal only): {len(self.train_data)}")
        vn = sum(not d["label"] for d in self.val_data)
        va = sum(d["label"]     for d in self.val_data)
        tn = sum(not d["label"] for d in self.test_data)
        ta = sum(d["label"]     for d in self.test_data)
        print(f"Validation: {len(self.val_data)} ({vn} normal, {va} anomalies)")
        print(f"Test:       {len(self.test_data)} ({tn} normal, {ta} anomalies)")


