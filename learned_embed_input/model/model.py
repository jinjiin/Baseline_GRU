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
        # self.embedding = nn.Embedding(1500, input_feature_size)
        self.gru = nn.GRU(input_feature_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, predict_len)


    def forward(self, x):
        # if self.embedding:
            # resume_path = '/lfs1/users/jbyu/NumberEmbedding/saved/models/TransE/0424_202416/model_best.pth'
            # checkpoint = torch.load(resume_path)
            # weight = checkpoint['state_dict']['entityEmbedding.weight']
            # embedding_entity = nn.Embedding.from_pretrained(weight)
            # # input = torch.LongTensor([1]).cuda()
            # # print(embedding(input))
            # # print(x.shape)
            # x = embedding_entity(x)
            # # print(x.shape)

        # x = self.embedding(x)
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
