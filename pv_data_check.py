#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
# import folium
import base64
# import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.utils.data as data

import random

import random as rand


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
# from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
# from skimage.io import imread, imshow
from sklearn.cluster import KMeans
# from torchsummary import summary as summary


# # 1. Os Local file 경로 설정
# 

# In[7]:


address = "./"
address_weather = "./data/weather_data/"
address_data = "./data/"
address_data_pv = "./data/pv_data/"


# In[83]:


EPS = 1e-5


# In[53]:


args = {
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCH': 20
}


# In[3]:


cd ..


# In[ ]:





# In[110]:





# In[5]:


# df_site_info = pd.read_csv(address+'data_each_PV_info_20191201_20210814.csv',index_col = 0)
# df_big_site_info = pd.read_csv(address+'geo_big_site_info.csv')


# In[8]:


df_sample = pd.read_csv(address_data+'pv_all_values.csv')
df_site_info = pd.read_csv(
    address_data+'data_each_PV_info_20191201_20210814(bell).csv', index_col=0)


# In[62]:


class PV_data(data.Dataset):
    def __init__(self, df_pv, df_info, transform):
        self.df_pv = df_pv
        self.df_info = df_info
        self.arr_pv_val = np.expand_dims(
            df_pv.values[:, 3:].T.reshape(-1, 89*7, 24), 3)
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
        return img_tensor


# In[84]:


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


# In[85]:


def plot_heat(arr):
    plt.pcolor(arr)
    plt.colorbar()
    plt.show()


# In[86]:


dataset = PV_data(df_sample, df_site_info, transform=ImageTransform())


# In[87]:


for idx in range(1):
    # A = ImageTransform(0,D[idx].max())
    A = ImageTransform()
    D = dataset.arr_pv_val[idx]
    B = A(D)
    plot_heat(D.squeeze())
    plot_heat(B.numpy()[0])


# In[88]:


train_dataset = PV_data(df_sample, df_site_info, transform=ImageTransform())

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size = args['BATCH_SIZE'],
    shuffle = True
)

# batch_iterator = iter(train_dataloader)
# input_img = next(batch_iterator)


# In[ ]:


# def build_XY(pv_num):
#     X = df_sample[pv_num].values
#     Y = df_site_info.loc[int(pv_num)]['bell']

#     return np.array([X, Y], dtype='object')


# df_cols = df_sample.columns.values[2:]
# XY = build_XY(df_cols[0])
# data_set = np.empty((1, 2), dtype='object')
# for i, col in enumerate(df_cols):
#     XY = build_XY(col)
#     data_set = np.insert(data_set, -1, XY, axis=0)
# data_set = np.delete(data_set, -1, axis=0)
# data_set[:, 1] = data_set[:, 1]*1

# # save_all_fig()

# for i in range(len(data_set)):
#     img = Image.open('img_pv/pv_{}_plt.png'.format(df_cols[i])).convert('L')
#     img_arr = np.array(img, dtype='uint8')
#     img_shape = img_arr.shape
#     img_arr = img_arr.reshape((1, img_shape[0], img_shape[1]))
#     data_set[i][0] = img_arr


# # Heatmap으로 Load Profile 시각화
# 

# In[51]:


# # train_set = torchvision.datasets.CIFAR10('./data', train=True,
# # 					download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['BATCH_SIZE'],
#                                            shuffle=True, num_workers=2)

# test_set = torchvision.datasets.CIFAR10('./data', train=False,
#                                         download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['BATCH_SIZE'],
#                                           shuffle=False, num_workers=2)


# classes = ['plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck']


# In[89]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[57]:


# class ConvAutoEncoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoEncoder, self).__init__()

#         # Encoder
#         self.cnn_layer1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))

#         self.cnn_layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))

#         # Decoder
#         self.tran_cnn_layer1 = nn.Sequential(
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
#             nn.ReLU())

#         self.tran_cnn_layer2 = nn.Sequential(
#             nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
#             nn.Sigmoid())

#     def forward(self, x):
#         output = self.cnn_layer1(x)
#         output = self.cnn_layer2(output)
#         output = self.tran_cnn_layer1(output)
#         output = self.tran_cnn_layer2(output)

#         return output


# In[107]:


class CAE_Network(nn.Module):
    def __init__(self, kernel_size, out_size, hidden_size):
        super(CAE_Network, self).__init__()
        self.k = kernel_size
        self.o = out_size
        self.h = hidden_size

        # Encoder Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.o, self.k, padding=1, stride=1),
            nn.BatchNorm2d(self.o),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.o, (self.o)**2, self.k, padding=1, stride=1),
            nn.BatchNorm2d((self.o)**2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Decoder Layers
        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d((self.o)**2, self.o,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.o, 1, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        print(x.shape)
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        latent = out
        out = self.trans_conv1(out)
        print(out.shape)
        out = self.trans_conv2(out)
        print(out.shape)
        
        return latent, out

    def detect(self, x):
        out = self.conv1(x)
        latent = self.conv2(out)

        return latent


# In[108]:


model = CAE_Network(3,4,64).to(device)
# model = ConvAutoEncoder()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])


# # Model Train

# In[109]:


steps = 0
total_steps = len(train_dataloader)
for epoch in range(args['NUM_EPOCH']):
    running_loss = 0
    for i, X_train in enumerate(train_dataloader):
        steps += 1
        X_train = X_train.to(device) ##
        _,output = model(X_train.float())
        loss = criterion(output, X_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()*X_train.shape[0]

        if steps % total_steps == 0:
            model.eval()
            print('Epoch: {}/{}'.format(epoch+1,
                  args['NUM_EPOCH']), 'Training Loss: {:.5f}..'.format(running_loss/total_steps))
            steps = 0
            running_loss = 0
            model.train()


# In[102]:


output.shape


# In[103]:


X_train.shape


# # 각 site의 daily curve를 겹쳐그린 image 저장

# In[16]:


site_set = ["seoul", "gangwon", "choongnam", "kwangjoo", "sejong", "busan", "daegoo",
            "choongbook", "daejeon", "woolsan", "incheon", "kssouth", "ksnorth", "kyoungki",
            "jn", "jb", "jeju"]


# In[17]:


weather_dict = {}
site_dict = {}
for k in range(len(site_set)):

    site = site_set[k]

    if site == "gangwon":
        site_korean = "강원"
        weather_korean = "강릉"
    elif site == "seoul":
        site_korean = "서울"
        weather_korean = "서울"
    elif site == "choongnam":
        site_korean = "충남"
        weather_korean = "천안"
    elif site == "choongbook":
        site_korean = "충북"
        weather_korean = "청주"
    elif site == "sejong":
        site_korean = "세종"
        weather_korean = "세종"
    elif site == "busan":
        site_korean = "부산"
        weather_korean = "부산"
    elif site == "kyoungki":
        site_korean = "경기"
        weather_korean = "수원"
    elif site == "kwangjoo":
        site_korean = "광주"
        weather_korean = "광주"
    elif site == "daegoo":
        site_korean = "대구"
        weather_korean = "대구"
    elif site == "daejeon":
        site_korean = "대전"
        weather_korean = "대전"
    elif site == "woolsan":
        site_korean = "울산"
        weather_korean = "울산"
    elif site == "incheon":
        site_korean = "인천"
        weather_korean = "인천"
    elif site == "kssouth":
        site_korean = "경남"
        weather_korean = "김해시"
    elif site == "ksnorth":
        site_korean = "경북"
        weather_korean = "경주시"
    elif site == "jn":
        site_korean = "전남"
        weather_korean = "장흥"
    elif site == "jb":
        site_korean = "전북"
        weather_korean = "장수"
    elif site == "jeju":
        site_korean = "제주"
        weather_korean = "제주"

    site_dict[site] = site_korean
    weather_dict[site_korean] = weather_korean


# In[25]:


list(site_dict.keys())


# In[26]:


for k in range(len(list(site_dict.keys()))):

    site = list(site_dict.keys())[k]
    site_korean = site_dict[site]
    weather_korean = weather_dict[site_korean]
    
    if not os.path.isdir(address+'test_plt_pv/{}'.format(site)):
        os.mkdir(address+'test_plt_pv/{}'.format(site))
        
    test_df_seoul = pd.read_csv(address+"new_PV_{}_value.csv".format(site),index_col = 0)
    pv_values = test_df_seoul.drop(["date","hour"],axis=1).values
    pv_values = pv_values.transpose().reshape(pv_values.shape[1],-1,24)
    pv_num_array = test_df_seoul.columns.values[2:]

    for i in range(pv_values.shape[0]):
        plt.figure(num=None, dpi=160)
        for j in range(360):
            plt.plot(pv_values[i][j],alpha=0.03)
            plt.title('Overlapped Curve(PV #{})'.format(pv_num_array[i]))
            plt.xlabel('Time(hour)')
            plt.ylabel('Power(Normalized)')
#             plt.axis('off')
#         plt.savefig(address+'test_plt_pv/{}/pv_{}_plt.png'.format(site,pv_num_array[i]))
        plt.show()
    


# In[9]:


from spectrum import Periodogram


# In[17]:


p = Periodogram(pv_values[0][2], sampling=1024)


# In[44]:


pv_values = test_df_seoul.drop(["date", "hour"], axis=1).values


# In[50]:


data = pv_values.transpose()


# In[71]:


for i in range(30):
    p = Periodogram(data[i], sampling=14952)
    p.run()
    p.plot()

    plt.rcParams['figure.figsize'] = [10, 6]
    plt.plot(p.psd)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


for i in range(30):
    plt.plot(pv_values[0][i], alpha=0.3)
    plt.title('Overlapped Curve(PV #{})'.format(pv_num_array[i]))
    plt.xlabel('Time(hour)')
    plt.ylabel('Power(Normalized)')
#     plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Curve 형태분석
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


site_set = ["seoul","gangwon","choongnam","kwangjoo","sejong","busan","daegoo","choongbook","daejeon","woolsan","incheon","kssouth","ksnorth","kyoungki","jn","jb","jeju"]

for k in range(len(site_set)):

    site = site_set[k]

    if site == "gangwon":
        site_korean = "강원"
        weather_korean = "강릉"
    elif site =="seoul":
        site_korean = "서울"
        weather_korean = "서울"
    elif site =="choongnam":
        site_korean = "충남"
        weather_korean = "천안"
    elif site =="choongbook":
        site_korean = "충북"
        weather_korean = "청주"
    elif site =="sejong":
        site_korean = "세종"
        weather_korean = "세종"
    elif site =="busan":
        site_korean = "부산"
        weather_korean = "부산"
    elif site =="kyoungki":
        site_korean = "경기"
        weather_korean = "수원"
    elif site =="kwangjoo":
        site_korean = "광주"
        weather_korean = "광주"
    elif site =="daegoo":
        site_korean = "대구"
        weather_korean = "대구"
    elif site =="daejeon":
        site_korean = "대전"
        weather_korean = "대전"
    elif site =="woolsan":
        site_korean = "울산"
        weather_korean = "울산"
    elif site =="incheon":
        site_korean = "인천"
        weather_korean = "인천"
    elif site =="kssouth":
        site_korean = "경남"
        weather_korean = "김해시"
    elif site =="ksnorth":
        site_korean = "경북"
        weather_korean = "경주시"
    elif site =="jn":
        site_korean = "전남"
        weather_korean = "장흥"
    elif site =="jb":
        site_korean = "전북"
        weather_korean = "장수"
    elif site =="jeju":
        site_korean = "제주"
        weather_korean = "제주"

    test_df_seoul = pd.read_csv(address+"new_PV_{}_value.csv".format(site),index_col = 0)
    pv_num_array = test_df_seoul.columns.values[2:]
    pca = PCA(n_components = 2)

    img_gray_list = []
    img_list = []

    for i in range(len(pv_num_array)):
        plt_img = imread('test_plt_pv/{}/pv_{}_plt.png'.format(site,pv_num_array[i]), 1)
        plt_img_rgb = imread('test_plt_pv/{}/pv_{}_plt.png'.format(site,pv_num_array[i]), 0)
        plt_pca = pca.fit_transform(plt_img)
        img_list.append(plt_img_rgb)
        img_gray_list.append(plt_pca.flatten())

    img_list = np.array(img_list)
    img_gray_list = np.array(img_gray_list)

    n_clusters = 2
    kmeans = KMeans(n_clusters=  n_clusters, random_state = 42).fit(img_gray_list)

    cluster_img_list=np.empty(2,dtype=object)
    cluster_num_list=np.empty(2,dtype=object)

    for n in range(n_clusters):
        cluster_img_list[n] = img_list[kmeans.labels_==n]
        cluster_num_list[n] = pv_num_array[kmeans.labels_==n]

    for n in range(n_clusters):
        if not os.path.isdir(address+'test_plt_pv/{}/cluster{}'.format(site,n+1)):
            os.mkdir(address+'test_plt_pv/{}/cluster{}'.format(site,n+1))
        for i in range(cluster_img_list[n].shape[0]):
            plt.figure(num=None, dpi=160)
            plt.title('Overlapped Curve(PV #{})'.format(cluster_num_list[n][i]))
            plt.xlabel('Time(hour)')
            plt.ylabel('Power(Normalized)')
#             plt.xlim([0, 24])      # X축의 범위: [xmin, xmax]

            imshow(cluster_img_list[n][i])
#             plt.axis('off')
            plt.savefig(address+'test_plt_pv/{}/cluster{}/curve_plt_{}.png'.format(site,n+1,cluster_num_list[n][i]))
        pd.DataFrame(cluster_num_list).to_csv(address+'test_plt_pv/{}/cluster{}/cluster_{}_isin.csv'.format(site,n+1,n+1))
        print("-"*100)
    


# In[11]:


cluster_img_list[0][1].shape


# In[10]:


site_set = ["kwangjoo", "sejong", "choongnam", "seoul", "busan", "daegoo", "choongbook",
            "daejeon", "woolsan", "incheon", "kssouth", "ksnorth", "kyoungki", "gangwon", "jn", "jb", "jeju"]

bell_list = np.array([], 'int')

for k in range(len(site_set)):

    site = site_set[k]

    if site == "gangwon":
        site_korean = "강원"
        weather_korean = "강릉"
    elif site == "seoul":
        site_korean = "서울"
        weather_korean = "서울"
    elif site == "choongnam":
        site_korean = "충남"
        weather_korean = "천안"
    elif site == "choongbook":
        site_korean = "충북"
        weather_korean = "청주"
    elif site == "sejong":
        site_korean = "세종"
        weather_korean = "세종"
    elif site == "busan":
        site_korean = "부산"
        weather_korean = "부산"
    elif site == "kyoungki":
        site_korean = "경기"
        weather_korean = "수원"
    elif site == "kwangjoo":
        site_korean = "광주"
        weather_korean = "광주"
    elif site == "daegoo":
        site_korean = "대구"
        weather_korean = "대구"
    elif site == "daejeon":
        site_korean = "대전"
        weather_korean = "대전"
    elif site == "woolsan":
        site_korean = "울산"
        weather_korean = "울산"
    elif site == "incheon":
        site_korean = "인천"
        weather_korean = "인천"
    elif site == "kssouth":
        site_korean = "경남"
        weather_korean = "김해시"
    elif site == "ksnorth":
        site_korean = "경북"
        weather_korean = "경주시"
    elif site == "jn":
        site_korean = "전남"
        weather_korean = "장흥"
    elif site == "jb":
        site_korean = "전북"
        weather_korean = "장수"
    elif site == "jeju":
        site_korean = "제주"
        weather_korean = "제주"

    site_list = os.listdir('test_plt_pv/{}/cluster1/'.format(site))[1:]

    for ele in site_list:
        bell_list = np.append(bell_list, int(ele[10:-4]))


# In[11]:


site_set = ["kwangjoo", "sejong", "choongnam", "seoul", "busan", "daegoo", "choongbook",
            "daejeon", "woolsan", "incheon", "kssouth", "ksnorth", "kyoungki", "gangwon", "jn", "jb", "jeju"]

non_bell_list = np.array([], 'int')

for k in range(len(site_set)):

    site = site_set[k]

    if site == "gangwon":
        site_korean = "강원"
        weather_korean = "강릉"
    elif site == "seoul":
        site_korean = "서울"
        weather_korean = "서울"
    elif site == "choongnam":
        site_korean = "충남"
        weather_korean = "천안"
    elif site == "choongbook":
        site_korean = "충북"
        weather_korean = "청주"
    elif site == "sejong":
        site_korean = "세종"
        weather_korean = "세종"
    elif site == "busan":
        site_korean = "부산"
        weather_korean = "부산"
    elif site == "kyoungki":
        site_korean = "경기"
        weather_korean = "수원"
    elif site == "kwangjoo":
        site_korean = "광주"
        weather_korean = "광주"
    elif site == "daegoo":
        site_korean = "대구"
        weather_korean = "대구"
    elif site == "daejeon":
        site_korean = "대전"
        weather_korean = "대전"
    elif site == "woolsan":
        site_korean = "울산"
        weather_korean = "울산"
    elif site == "incheon":
        site_korean = "인천"
        weather_korean = "인천"
    elif site == "kssouth":
        site_korean = "경남"
        weather_korean = "김해시"
    elif site == "ksnorth":
        site_korean = "경북"
        weather_korean = "경주시"
    elif site == "jn":
        site_korean = "전남"
        weather_korean = "장흥"
    elif site == "jb":
        site_korean = "전북"
        weather_korean = "장수"
    elif site == "jeju":
        site_korean = "제주"
        weather_korean = "제주"

    site_list = os.listdir('test_plt_pv/{}/cluster2/'.format(site))[1:]

    for ele in site_list:
        non_bell_list = np.append(non_bell_list, int(ele[10:-4]))


# In[12]:


os.listdir('test_plt_pv/{}/cluster1/'.format(site))[1:]


# In[13]:


info = pd.read_csv('data_each_PV_info_20191201_20210814.csv')


# In[14]:


info['bell'] = 0


# In[15]:


k = 0
for ele in bell_list:
    #     print(k)
    #     k+=1
    #     print(info.loc[info['번호'] == ele,'bell'])
    info.loc[info['번호'] == ele, 'bell'] = True


# In[16]:


k = 0
for ele in non_bell_list:
    #     print(k)
    #     k+=1
    #     print(info.loc[info['번호'] == ele,'bell'])
    info.loc[info['번호'] == ele, 'bell'] = False


# In[17]:


info


# In[18]:


info = info.drop('Unnamed: 0', axis=1)


# In[19]:


info.to_csv('data_each_PV_info(test_bell).csv',
            encoding='utf-8-sig', index=False)


# In[29]:


test_info = pd.read_csv('data_each_PV_info(test_bell).csv', index_col=0)


# In[30]:


test_info = test_info.drop(['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12',
                            'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15',
                            'Unnamed: 16', 'Unnamed: 17'], axis=1)


# In[31]:


test_info


# In[ ]:


site_set = ["kwangjoo", "sejong", "choongnam", "seoul", "busan", "daegoo", "choongbook",
            "daejeon", "woolsan", "incheon", "kssouth", "ksnorth", "kyoungki", "gangwon", "jn", "jb", "jeju"]

for k in range(len(site_set)):

    site = site_set[k]

    if site == "gangwon":
        site_korean = "강원"
        weather_korean = "강릉"
    elif site == "seoul":
        site_korean = "서울"
        weather_korean = "서울"
    elif site == "choongnam":
        site_korean = "충남"
        weather_korean = "천안"
    elif site == "choongbook":
        site_korean = "충북"
        weather_korean = "청주"
    elif site == "sejong":
        site_korean = "세종"
        weather_korean = "세종"
    elif site == "busan":
        site_korean = "부산"
        weather_korean = "부산"
    elif site == "kyoungki":
        site_korean = "경기"
        weather_korean = "수원"
    elif site == "kwangjoo":
        site_korean = "광주"
        weather_korean = "광주"
    elif site == "daegoo":
        site_korean = "대구"
        weather_korean = "대구"
    elif site == "daejeon":
        site_korean = "대전"
        weather_korean = "대전"
    elif site == "woolsan":
        site_korean = "울산"
        weather_korean = "울산"
    elif site == "incheon":
        site_korean = "인천"
        weather_korean = "인천"
    elif site == "kssouth":
        site_korean = "경남"
        weather_korean = "김해시"
    elif site == "ksnorth":
        site_korean = "경북"
        weather_korean = "경주시"
    elif site == "jn":
        site_korean = "전남"
        weather_korean = "장흥"
    elif site == "jb":
        site_korean = "전북"
        weather_korean = "장수"
    elif site == "jeju":
        site_korean = "제주"
        weather_korean = "제주"

    print("---------- {} ----------".format(site_korean))
    site_bell_info = test_info[test_info['광역지역']
                               == site_korean][['bell', 'label']]
    num_t


# In[38]:


site_bell_info = test_info[test_info['광역지역'] == "제주"][['bell', 'label']]


# In[48]:


site_tf_arr = site_bell_info.values[:, 0] == site_bell_info.values[:, 1]


# In[49]:


np.unique(site_tf_arr, return_counts=True)


# In[50]:


np.average(site_tf_arr)


# In[ ]:




