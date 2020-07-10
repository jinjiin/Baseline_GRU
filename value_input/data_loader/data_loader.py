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
        self.feature = torch.FloatTensor(np.load(feature_path))
        self.target = np.load(target_path)

        # self.feature_mean = torch.mean(self.feature, dim=0)
        # self.feature_std = torch.std(self.feature, dim=0)
        # self.feature = (self.feature - self.feature_mean) / self.feature_std
        # print(self.feature.shape)

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        return self.feature[idx], torch.FloatTensor(self.target[idx])


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
