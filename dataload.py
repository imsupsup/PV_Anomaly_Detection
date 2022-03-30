import pandas as pd
import numpy as np
from hyper import *
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms




df_sample = pd.read_csv(address_data+'pv_all_values.csv')
df_site_info = pd.read_csv(
    address_data+'data_each_PV_info_20191201_20210814(bell).csv', index_col=0)



class PV_data(data.Dataset):
    def __init__(self, df_pv, df_info, transform):
        self.df_pv = df_pv
        self.df_info = df_info
        self.arr_pv_val = np.expand_dims(
            df_pv.values[:14952-72][:, 3:].T.reshape(-1, W+4, H), 3)
        self.arr_pv_num = df_pv.columns.values[3:]
        self.transform = transform

    def is_bell(self, num):
        bell = self.df_info.loc[int(num)]['bell']
        return bell

    def pv_num(self, idx):
        num = int(self.arr_pv_num[idx])
        return num

    def pv_idx(self, pv_num):
        idx = np.where(self.arr_pv_num == str(pv_num))[0][0]
        return idx

    def value(self, num):
        val = self.df_pv[num].values
        return val

    def __len__(self):
        return len(self.arr_pv_val)
    
    def __getitem__(self, idx):
        img_tensor = self.transform(self.arr_pv_val[idx])
        return img_tensor, self.is_bell(self.pv_num(idx))


class PV_data_3layer(data.Dataset):
    def __init__(self, df_pv, df_info, transform):
        self.df_pv = df_pv
        self.df_info = df_info
        self.arr_pv_val = np.expand_dims(
            df_pv.values[:14952-24*7][:, 3:].T.reshape(-1, W, H), 3)
        self.arr_pv_num = df_pv.columns.values[3:]
        self.transform = transform

    def is_bell(self, num):
        bell = self.df_info.loc[int(num)]['bell']
        return bell

    def pv_num(self, idx):
        num = int(self.arr_pv_num[idx])
        return num

    def pv_idx(self, pv_num):
        idx = np.where(self.arr_pv_num == str(pv_num))[0][0]
        return idx

    def value(self, num):
        val = self.df_pv[num].values
        return val

    def __len__(self):
        return len(self.arr_pv_val)
    
    def __getitem__(self, idx):
        img_tensor = self.transform(self.arr_pv_val[idx])
        return img_tensor, self.is_bell(self.pv_num(idx))



class ImageTransform():

    def __init__(self):
        pass

    def __call__(self, img):
        mean = 0
        std = img.max()
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean+EPS, std+EPS)
        ])
        return self.data_transform(img)







train_dataset =  PV_data(df_sample, df_site_info, transform=ImageTransform())

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size = args['BATCH_SIZE'],
    shuffle = True
)









