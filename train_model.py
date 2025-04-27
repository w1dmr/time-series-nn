import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from rnn import RNN


def train_model(d_train, train_data, epochs, batch_size, learning_rate, weight_decay, hidden_size, lr_scheduler_gamma, device):
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = RNN(d_train.in_features, d_train.out_features, hidden_size)
    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_scheduler_gamma)

    mse = nn.MSELoss()

    model.train()
    for _e in range(epochs):
        loss_mean = 0
        lm_count = 0

        train_tqdm = tqdm(train_data, leave=True, ncols=100)
        for x_train, y_train in train_tqdm:
            x_train, y_train = x_train.to(device), y_train.to(device)

            predict = model(x_train).squeeze(0)
            loss = mse(predict, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lm_count += 1
            loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
            train_tqdm.set_description(
                f'Epoch [{_e + 1} / {epochs}], LR: {optimizer.param_groups[0]["lr"]:.5f}, loss_mean = {loss_mean:.3f}')

        scheduler.step()

    return model
