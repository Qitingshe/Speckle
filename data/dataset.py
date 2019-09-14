from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch as t
import numpy as np


class IMGDataset(data.Dataset):
    def __init__(self, imgs, train=True, test=False):
        # root :image rootdir
        self.test = test

        # get the dataset,img为路径下的图片名称，imgs为图片的路径

        # get sample number of the dataset
        imgs_num = len(imgs)
        # random.shuffle(imgs)

        if self.test:
            self.imgs = imgs
        # 分割train为训练集和验证集，比例为7：3
        elif train:
            # 前70%的数据作为trainset(注意中括号中的冒号)
            # self.imgs=imgs[:int(0.7*imgs_num)]
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            # 后30%的数据作为valset
            self.imgs = imgs[int(0.7*imgs_num):]
            # self.imgs = imgs
        # 数据预处理

        # 标准化至[-1,1],同时规定均值和标准差
        normalize = T.Normalize(mean=[0.485, ],
                                std=[0.229, ])

        # 使用Compose将对数据的处理操作拼接起来，和nn.Sequential相似，该操作以对象的形式存在，需要调用他的__call__方法
        self.transforms = T.Compose([
            T.CenterCrop(512),     # 从图片中裁剪出224×224的图片
            # T.RandomVerticalFlip(),
            # T.RandomHorizontalFlip(),
            # T.RandomCrop(224),
            T.Resize(224),
            T.ToTensor(),       # 将图片转成Tensor，归一化至[0,1]
            normalize           # 标准化至[-1,1]，规定均值和方差
        ])

    def __getitem__(self, index):
        """
        将文件读取等费时操作放在__getitem__函数中，利用多线程加速。一次调用该函数，只返回一个样本，在多进程中会并行地调用__getitem__函数，由此实现加速
        如果需要使用batch，打乱以及并行加速等操作，需要继续使用PyTorch的DataLoader
        DataLoader是一个可迭代对象，所以我们可以在for循环中使用它
        """
        img_path = self.imgs[index]
        # 获取图片的label
        name = img_path.split('.')[-2].split('/')[-1]
        label = list(name)
        label = np.asarray(label, dtype=np.uint8)
        label = t.LongTensor(label)
        # 获取数据
        data = Image.open(img_path)
        # 数据预处理
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
