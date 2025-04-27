import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.LSTM(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, out_features)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        # hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.out(h)
        return y
