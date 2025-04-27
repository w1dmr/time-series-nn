import os

import numpy as np
import torch
import torch.utils.data as data


class TimeSeriesDatasetForRNN(data.Dataset):
    def __init__(self, path, window_size, split, train_ratio, normalize_targets,
                 kfold_split, kfold_train_idxs, kfold_val_idxs):
        self.path = path
        self.window_size = window_size
        self.in_features = self.out_features = 4
        self.split = split
        self.train_ratio = train_ratio
        self.normalize_targets = normalize_targets

        # Параметры для кросс-валидации
        self.kfold_split = kfold_split
        self.kfold_train_idxs = kfold_train_idxs
        self.kfold_val_idxs = kfold_val_idxs

        self.seed = 42

        all_files = os.listdir(self.path)
        all_files = np.array(all_files)

        np.random.seed(self.seed)
        np.random.shuffle(all_files)

        if self.train_ratio is not None:
            split_idx = int(len(all_files) * self.train_ratio)

        if kfold_split and self.kfold_train_idxs is not None and self.split == 'train':  # Если используется кросс-валидация
            self.list_files = all_files[self.kfold_train_idxs]
        elif kfold_split and self.kfold_val_idxs is not None and self.split == 'val':
            self.list_files = all_files[self.kfold_val_idxs]
        elif not self.kfold_split and self.split == 'train':
            self.list_files = all_files[:split_idx]
        elif not self.kfold_split and self.split == 'test':
            self.list_files = all_files[split_idx:]

        self.data = self._load_data()
        self.X, self.y = self._create_windows()

        if self.split == 'train':
            self.x_mean = self.X.mean(dim=(0, 1))
            self.x_std = self.X.std(dim=(0, 1)) + 1e-8

            if self.normalize_targets:
                self.y_mean = self.y.mean(dim=0)
                self.y_std = self.y.std(dim=0) + 1e-8
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None

    def set_normalization_params(self, x_mean, x_std, y_mean=None, y_std=None):
        self.x_mean = x_mean
        self.x_std = x_std
        self.X = (self.X - self.x_mean) / self.x_std

        if self.normalize_targets and y_mean is not None and y_std is not None:
            self.y_mean = y_mean
            self.y_std = y_std
            self.y = (self.y - self.y_mean) / self.y_std

    def denormalize(self, normalize_features=True, normalize_targets=True):
        if normalize_features and self.x_mean is not None and self.x_std is not None:
            self.X = self.X * self.x_std + self.x_mean

        if normalize_targets and self.y_mean is not None and self.y_std is not None:
            self.y = self.y * self.y_std + self.y_mean

    def _load_data(self):
        all_series = []
        for file_name in self.list_files:
            dataset = np.loadtxt(os.path.join(self.path, file_name), dtype=np.float32, delimiter=' ')[:, 5:]
            all_series.append(dataset)
        return all_series

    def _create_windows(self):
        X_list = []
        y_list = []

        for series in self.data:
            length = len(series)
            for i in range(length - self.window_size + 1):
                window = series[i:i + self.window_size, :self.in_features]
                target = series[i][self.in_features:]
                X_list.append(window)
                y_list.append(target)

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        y = torch.tensor(np.stack(y_list), dtype=torch.float32)
        return X, y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    from sklearn.model_selection import KFold

    path = 'datasets'
    all_files = os.listdir(path)
    num_folds = 16
    kf = KFold(n_splits=num_folds, shuffle=False)

    for train_idxs, val_idxs in kf.split(all_files):
        d_train = TimeSeriesDatasetForRNN(path=path, split='train', kfold_split=True, kfold_train_idxs=train_idxs)
        d_val = TimeSeriesDatasetForRNN(path=path, split='val', kfold_split=True, kfold_val_idxs=val_idxs)
