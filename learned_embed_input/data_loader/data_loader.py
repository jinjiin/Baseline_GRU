import torch.utils.data as data
import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
import torch


class BaseDataset(data.Dataset):
    def __init__(self, mode, time_len, embed_path="embed_128_no_l2.npy", feature_choice='pm25'):
        super(BaseDataset).__init__()

        # mode:train/valid
        # time_len:6/12/24/48
        # embed_path：embed_{}_no_l2.npy：50\64\128\256\512，no_l2\l2
        # feature_choice:pm25/pm10

        self.mode = mode
        self.feature_choice = feature_choice
        feature_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_Pm25_Pm10_feature_hour{}.npy'.format(mode, time_len)
        target_path = '/lfs1/users/jbyu/QA_baseline/np_data/{}_Pm25_Pm10_target_hour{}.npy'.format(mode, time_len)
        self.embed_path = embed_path
        self.feature = np.load(feature_path)
        self.target = np.load(target_path)

        print(self.feature.shape)

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        # seperate embedding
        embed_path = '/lfs1/users/jbyu/NumberEmbedding_0_99/saved_embedding/' + self.embed_path
        # print(embed_path)
        embedding_entity = np.load(embed_path)
        embedding_entity = torch.FloatTensor(embedding_entity)  # .cuda()

        # if self.feature_choice == 'pm25':
        #     pm25_feature = embedding_entity.index_select(0, torch.LongTensor(self.feature[idx][:, 0]))
        #     target = self.target[idx][:, 0]
        #     return torch.FloatTensor(pm25_feature), torch.FloatTensor(target)
        # else:
        #     pm10_feature = embedding_entity.index_select(0, torch.LongTensor(self.feature[idx][:, 1]))
        #     target = self.target[idx][:, 1]
        #     return torch.FloatTensor(pm10_feature), torch.FloatTensor(target)

        # combination embedding
        pm25_feature = embedding_entity.index_select(0, torch.LongTensor(self.feature[idx][:, 0]))
        pm10_feature = embedding_entity.index_select(0, torch.LongTensor(self.feature[idx][:, 1]))
        target = self.target[idx]
        feature = torch.cat((pm25_feature, pm10_feature), dim=1)
        return feature, torch.FloatTensor(target)



class dataLoader(DataLoader):
    def __init__(self, mode, time_len, embed_path, feature_choice, batch_size, num_workeres=1, shuffle=True):
        self.dataset = BaseDataset(mode, time_len, embed_path, feature_choice)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workeres)


if __name__ == '__main__':
    valid_loader = dataLoader(
                                mode='train',
                                time_len=24,
                                embed_path="embed_50_no_l2.npy",
                                feature_choice='pm25',
                                batch_size=512,
                                num_workeres=1,
                                shuffle=True)

    for batch_idx, (feature, target) in enumerate(valid_loader):
        print('-------------------------')
        print(feature.shape, target.shape)
