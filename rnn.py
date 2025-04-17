import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from dataset_for_rnn import TimeSeriesDatasetForRNN


class RNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 32
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        # hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.out(h)
        return y


seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d_train = TimeSeriesDatasetForRNN('datasets')
d_train.set_normalization_params(d_train.x_mean, d_train.x_std)
train_data = data.DataLoader(d_train, batch_size=64, shuffle=True)

model = RNN(d_train.in_features, d_train.out_features)

model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.001)
mae = nn.L1Loss()
smape = lambda output, target: torch.mean(
    100 * torch.abs(target - output) / (torch.abs(target) + torch.abs(output)))
mse = nn.MSELoss()

epochs = 10

model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True, ncols=100)
    for x_train, y_train in train_tqdm:
        x_train, y_train = x_train.to(device), y_train.to(device)

        # predict = model(x_train)
        predict = model(x_train).squeeze(0)
        loss = mse(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f'Epoch [{_e + 1} / {epochs}], loss_mean = {loss_mean:.3f}')

st = model.state_dict()
torch.save(st, 'model_rnn.tar')

d_test = TimeSeriesDatasetForRNN('datasets', split='test')
d_test.set_normalization_params(d_train.x_mean, d_train.x_std)
test_data = data.DataLoader(d_test, shuffle=False)

Q_mae = Q_smape = Q_mse = Q_rmse = 0
all_preds = []
all_targets = []

model.eval()

for x_test, y_test in test_data:
    with torch.no_grad():
        x_test, y_test = x_test.to(device), y_test.to(device)
        # predict = model(x_test)
        predict = model(x_test).squeeze(0)

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
