import torch.utils.data as data
import PIL.Image as Image

import glob
import os
import random


class MakeDataset(data.Dataset):
    def __init__(self, dataset_dir, data_folder, cls_folder, transforms):
        """
        Custom Dataset params

        :param dataset_dir: DATABASE PATH
        :param data_folder: Dataset folder A or B
        :param cls_folder: Image classes folder
        :param transforms: Image style transform module
        """
        super(MakeDataset, self).__init__()
        self.dataset_dir = dataset_dir

        folder_A = glob.glob(f'{os.path.join(dataset_dir, data_folder, cls_folder[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, data_folder, cls_folder[1])}/*')
        # print(folder_B)

        self.transform = transforms

        self.image_path = []

        for i in range(len(folder_A)):
            self.image_path.append([folder_A[i], 0])

        for i in range(len(folder_B)):
            self.image_path.append([folder_B[i], 1])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        item = self.transform(Image.open(self.image_path[index][0]))
        label = self.image_path[index][1]

        return [item, label]