import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')



def pc_normalize(pc): 
    """
    数据归一化到单位球
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class TeethDataLoader(Dataset):
    def __init__(self, split='train', cache_size=15000):
        # 数据集目录
        self.root = "data_1000/teeth_data/" 

        teethFiles = {} #字典
        teethFiles['train'] = [line.rstrip() for line in open(self.root + 'teeth_train.txt')] #训练样本
        teethFiles['test'] = [line.rstrip() for line in open(self.root + 'teeth_test.txt')] #测试样本

        #生成文件路径的list
        self.datapath = [self.root + teethFiles[split][i] + '.txt' for i in range(len(teethFiles[split]))] 
        #打印训练集或是测试集的大小，文件数目
        print('The size of %s data is %d'%(split,len(self.datapath))) 

        # 缓存多少数据到内存中
        self.cache_size = cache_size  
        self.cache = {} 

    def __getitem__(self, index):
        if index in self.cache:
            point_set, features = self.cache[index]
        else:
            f = self.datapath[index]
            # 取出数据
            data = np.loadtxt(f)
            # 取出坐标信息
            point_set = data[:, 0:3]
             #取出groundtruth的heatmap信息,分别是CO, CU, FA, OC
            features = data[:, 3:7]
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, features) 
        # 归一化
        point_set = pc_normalize(point_set) 
        return point_set, features

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    import torch

    data = TeethDataLoader(split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    for point,features in DataLoader:
        print(point.shape)
        print(features.shape)