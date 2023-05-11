import torch
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms as torch_transformations
from torch.utils.data import DataLoader, Dataset
import cv2
import os
from torchvision import transforms as Tranformations
import albumentations as Augmentations
from os import listdir


class DataSet(Dataset):
    """
    Custom dataset class, to provide interface between torch dataloader and our dataset
    """

    def __init__(self, image_path, mask_path, mean, std, transform=None):
        """
        Constructor for dataset object.
        :param img_path: path to images directory
        :param mask_path: path to mask directory
        :param mean: Image mean(per channel)
        :param std: Image standard deviation(per channel)
        :param transform: Transformation
        """
        self.img_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.mean = mean
        self.std = std
        self.file_list = os.listdir(self.img_path)

    def __len__(self):
        """
        Returns the number if samples within the dataset
        :return: number of samples
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Interface between torch dataloader and our dataset
        :param index: index of image to retrieve
        :return: Image, Mask
        """
        image = (cv2.imread(self.img_path + "/" + self.file_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.mask_path + "/" + self.file_list[index]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = Image.fromarray(aug['image'])
            mask = aug['mask']

        if self.transform is None:
            image = Image.fromarray(image)

        t = Tranformations.Compose([Tranformations.ToTensor(), Tranformations.Normalize(self.mean, self.std),
                                    Tranformations.Resize(size=(256, 256))])
        image = t(image)
        mask = torch.from_numpy(mask).long()
        return image, mask

    def return_file_list(self):
        """
        Returns the names of the files within the dataset, images only
        :return: list of file names
        """
        return self.file_list
