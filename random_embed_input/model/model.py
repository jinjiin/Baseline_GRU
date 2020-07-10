import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GRU_Base(nn.Module):
    def __init__(self, input_feature_size=18, hidden_size=512, predict_len=8, num_layers=2, dropout=0.0):
        super().__init__()

        """
        input_size:length of feature_attribute
        """
        self.embedding = nn.Embedding(1500, input_feature_size)
        self.gru = nn.GRU(input_feature_size*2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_len)

    def forward(self, x):
        # each number out
        # x = self.embedding(x)


        # combination out
        x1, x2 = torch.chunk(x, 2, dim=2)
        x1 = x1.squeeze(2)
        x2 = x2.squeeze(2)
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=2)


        lstm_out, _ = self.gru(x)
        return self.linear(lstm_out[:, -1, :])


if __name__ == '__main__':
    batchsize = 10
    seq_len = 24
    context_feature = 1

    x = np.random.randint(10,size=(batchsize, seq_len, context_feature))
    x = torch.LongTensor(x)
    # context = torch.randn((batchsize, context_feature, seq_len))
    model = GRU_Base(input_feature_size=context_feature, hidden_size=32, predict_len=1, num_layers=2, dropout=0.0, embedding=True)
    res = model(x)
    print(res.shape)
