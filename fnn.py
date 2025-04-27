import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from dataset_for_fnn import TimeSeriesDatasetForFNN

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_train = TimeSeriesDatasetForFNN('datasets', Q_window=False, Q_lags=4, n_window=True, n_lags=4, filled=False)
d_train.set_normalization_params(d_train.x_mean, d_train.x_std)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(16, 8),
    nn.Tanh(),
    nn.Linear(8, 4),
)

model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
mae = nn.L1Loss()  # MAE
smape = lambda output, target: torch.mean(
    100 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output)))  # SMAPE
mse = nn.MSELoss()  # MSE

epochs = 15

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

d_test = TimeSeriesDatasetForFNN('datasets', Q_window=False, Q_lags=4, n_window=True, n_lags=4, filled=False,
                                 split='test')
d_test.set_normalization_params(d_train.x_mean, d_train.x_std)
test_data = data.DataLoader(d_test, shuffle=False)

out_dim = 4
mae_vals = []
smape_vals = []
mse_vals = []

model.eval()

for i, (x_test, y_test) in enumerate(test_data):
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        predict = model(x_test)

        if i % 365 == 0:
            print(f"Target: {y_test.cpu().numpy()}")
            print(f"Predict: {predict.cpu().numpy()}")

        mae_vals.append(torch.abs(predict - y_test).cpu().numpy())  # MAE отдельно по всем параметрам
        smape_vals.append((100 * torch.abs(predict - y_test) / (
                torch.abs(predict) + torch.abs(
            y_test))).cpu().numpy())  # SMAPE отдельно по всем параметрам
        mse_vals.append(((predict - y_test) ** 2).cpu().numpy())  # MSE отдельно по всем параметрам

mae_vals = np.concatenate(mae_vals, axis=0)
smape_vals = np.concatenate(smape_vals, axis=0)
mse_vals = np.concatenate(mse_vals, axis=0)
rmse_vals = np.sqrt(mse_vals)

mean_mae_per_feature = np.mean(mae_vals, axis=0)
mean_smape_per_feature = np.mean(smape_vals, axis=0)
mean_mse_per_feature = np.mean(mse_vals, axis=0)
mean_rmse_per_feature = np.mean(rmse_vals, axis=0)

print('Ошибки по каждому выходному параметру:')
print(f'MAE = {mean_mae_per_feature}')
print(f'SMAPE = {mean_smape_per_feature}')
print(f'MSE = {mean_mse_per_feature}')
print(f'RMSE = {mean_rmse_per_feature}')
