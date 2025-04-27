import csv
import os

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import KFold

from dataset_for_rnn import TimeSeriesDatasetForRNN
from test_model import test_model
from train_model import train_model

# Гиперпараметры
WINDOW_SIZE = 2
NORMALIZE_TARGETS = False
EPOCHS = 10
BATCH_SIZE = 29
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
HIDDEN_SIZE = 13
GAMMA = 0.95
NUM_FOLDS = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

kf = KFold(n_splits=NUM_FOLDS)
all_files = os.listdir('datasets')

results = []

for i, (train_idxs, val_idxs) in enumerate(kf.split(all_files)):
    d_train = TimeSeriesDatasetForRNN(path='datasets',
                                      window_size=WINDOW_SIZE,
                                      split='train',
                                      train_ratio=0.8,
                                      normalize_targets=NORMALIZE_TARGETS,
                                      kfold_split=True,
                                      kfold_train_idxs=train_idxs,
                                      kfold_val_idxs=None)

    d_val = TimeSeriesDatasetForRNN(path='datasets',
                                    window_size=WINDOW_SIZE,
                                    split='val',
                                    train_ratio=None,
                                    normalize_targets=NORMALIZE_TARGETS,
                                    kfold_split=True,
                                    kfold_train_idxs=None,
                                    kfold_val_idxs=val_idxs)

    if NORMALIZE_TARGETS:
        d_train.set_normalization_params(d_train.x_mean, d_train.x_std, d_train.y_mean, d_train.y_std)
        d_val.set_normalization_params(d_train.x_mean, d_train.x_std, d_train.y_mean, d_train.y_std)
    else:
        d_train.set_normalization_params(d_train.x_mean, d_train.x_std)
        d_val.set_normalization_params(d_train.x_mean, d_train.x_std)

    train_data = data.DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
    val_data = data.DataLoader(d_val, batch_size=BATCH_SIZE, shuffle=False)

    # Обучение модели на текущем фолде
    model = train_model(d_train=d_train,
                        train_data=train_data,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        hidden_size=HIDDEN_SIZE,
                        lr_scheduler_gamma=GAMMA,
                        device=DEVICE)

    mae_results, smape_results = test_model(d_train=d_train,
                                            d_test=d_val,
                                            test_data=val_data,
                                            model=model,
                                            device=DEVICE)

    print(f'Fold {i + 1}:')
    print(mae_results)
    print(smape_results)
    print(d_train.list_files)

    results.append(np.concatenate((mae_results, smape_results)))

with open('results.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(results)
