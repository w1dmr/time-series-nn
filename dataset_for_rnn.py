import os

import numpy as np
import torch
import torch.utils.data as data


class TimeSeriesDatasetForRNN(data.Dataset):
    def __init__(self, path, window_size=5, split='train', train_ratio=0.8):
        self.path = path
        self.window_size = window_size
        self.in_features = self.out_features = 4
        self.split = split
        self.train_ratio = train_ratio
        self.seed = 42

        all_files = os.listdir(self.path)
        np.random.seed(self.seed)
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * self.train_ratio)

        if self.split == 'train':
            self.list_files = all_files[:split_idx]
        elif self.split == 'test':
            self.list_files = all_files[split_idx:]
        else:
            self.list_files = all_files

        self.data = self._load_data()
        self.X, self.y = self._create_windows()

        if self.split == 'train':
            self.x_mean = self.X.mean(dim=(0, 1))
            self.x_std = self.X.std(dim=(0, 1)) + 1e-8
        else:
            self.x_mean = None
            self.x_std = None

    def set_normalization_params(self, x_mean, x_std):
        self.x_mean = x_mean
        self.x_std = x_std

        self.X = (self.X - self.x_mean) / self.x_std

    def denormalize(self):
        self.X = self.X * self.x_std + self.x_mean

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
    d_train = TimeSeriesDatasetForRNN(path='datasets', window_size=5)
    print(d_train[0])
    d_train.set_normalization_params(d_train.x_mean, d_train.x_std)
    print(d_train[0])
    d_train.denormalize()
    print(d_train[0])