import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from dataset_for_fnn import TimeSeriesDatasetForFNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_train = TimeSeriesDatasetForFNN('datasets', Q_window=True, Q_lags=4, n_window=True, n_lags=4, filled=False)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(20, 9),
    nn.Dropout(0.3),
    nn.Tanh(),
    nn.Linear(9, 4),
)

model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
mae = nn.L1Loss()  # MAE
smape = lambda output, target: torch.mean(
    100 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output)))  # SMAPE
mse = nn.MSELoss()  # MSE

epochs = 20

model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True, ncols=100)
    for x_train, y_train in train_tqdm:
        x_train, y_train = x_train.to(device), y_train.to(device)

        predict = model(x_train)
        loss = mse(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f'Epoch [{_e + 1} / {epochs}], loss_mean = {loss_mean:.3f}')

d_test = TimeSeriesDatasetForFNN('datasets', Q_window=True, Q_lags=4, n_window=True, n_lags=4, filled=False, split='test')
test_data = data.DataLoader(d_test, shuffle=False)

Q_mae = Q_smape = Q_mse = Q_rmse = 0
all_preds = []
all_targets = []

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        predict = model(x_test)

        Q_mae += mae(predict, y_test).item()
        Q_smape += smape(predict, y_test).item()
        Q_mse += mse(predict, y_test).item()

        all_preds.append(predict.cpu().numpy())
        all_targets.append(y_test.cpu().numpy())

Q_mae /= len(d_test)
Q_smape /= len(d_test)
Q_mse /= len(d_test)
Q_rmse = np.sqrt(Q_mse)

print(f'\nMAE на тестовой выборке: {Q_mae:.5f}')
print(f'SMAPE на тестовой выборке: {Q_smape:.2f}%')
print(f'MSE на тестовой выборке: {Q_mse:.5f}')
print(f'RMSE на тестовой выборке: {Q_rmse:.5f}')

all_preds_np = np.concatenate(all_preds, axis=0)
all_targets_np = np.concatenate(all_targets, axis=0)

print(all_preds_np)
print(all_targets_np)