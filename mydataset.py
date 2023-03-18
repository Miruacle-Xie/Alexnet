import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, npy_file, type):
        self.data = np.load(npy_file)
        self.type = type
        if self.type == "train":
            self.images = self.data[:, :-1].reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0
            self.labels = self.data[:, -1].astype(np.int64)
        elif type == "test":
            print(self.data)
        #     self.images = self.data[:, :].reshape(3, 32, 32).transpose((1, 2, 0)).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.type == "train":
            image = self.images[index]
            label = self.labels[index]
            return torch.from_numpy(image), torch.tensor(label)
        elif self.type == "test":
            image = self.data[index].reshape(3, 32, 32)
            image = image.transpose((1, 2, 0))  # 转换通道顺序为 (height, width, channel)
            image = image.astype(np.float32) / 255.0  # 将像素值缩放到 [0, 1] 范围内
            return image




def main():

    # data_path = os.path.abspath(os.path.join("D:\\VScode\\Datasets\\flower"))
    train_dataset = MyDataset("data_train.npy", "train")
    vaild_dataset = MyDataset("data_test_x.npy", "test")
    # train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    # vaild_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["test"])
    train_num = len(train_dataset)
    test_num = len(vaild_dataset)
 


if __name__ == '__main__':
    main()
