"""
2020-03-17 작성
Kang Jun Hyeok

딥러닝 입력을 제공할 dataset class 생성
이미지를 불러와서 perturbator 진행 후 반환
"""


import os
import sys

import random
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from perturbator.binarization_perturbator import BinarizationPerturbator
from perturbator.blackout_perturbator import BlackoutPerturbator
from perturbator.edge_detection_perturbator import EdgeDetectionPerturbator
from perturbator.smoothing_perturbator import SmoothingPerturbator
from perturbator.whiteout_perturbator import WhiteoutPerturbator
from my_transforms import my_transforms
from data_info.data_info import DataInfo
from my_enums.dataset_enum import DatasetEnum
from my_enums.perturbator_enum import PerturbatorEnum


class MyDataset(Dataset):
    def __init__(self, dataset_type=DatasetEnum.TEST1, perturbation_ratio=None, aug=True):

        self.init_trans = transforms.Compose([transforms.Resize((DataInfo.resized_image_size[1], DataInfo.resized_image_size[0])),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])

        self.col_trans = my_transforms.Compose([my_transforms.ToPILImage(),
                                           my_transforms.ColorJitter(brightness=1,
                                                                     contrast=1,
                                                                     saturation=1,
                                                                     hue=0.5
                                                                     ),
                                           my_transforms.ToTensor(),
                                           ])

        self.dataset_type = dataset_type
        self.input_image_folder = torchvision.datasets.ImageFolder(
            root=os.path.join(DataInfo.train_test_image_folder_path, self.dataset_type.name.lower(), 'input'),
            transform=self.init_trans)
        self.heatmap_image_folder = torchvision.datasets.ImageFolder(
            root=os.path.join(DataInfo.train_test_image_folder_path, self.dataset_type.name.lower(), 'heatmap'),
            transform=self.init_trans)
        self.data_num = len(self.input_image_folder)
        self.aug = aug
        self.width = DataInfo.resized_image_size[0]
        self.height = DataInfo.resized_image_size[1]
        self.scheme = DataInfo.perturbation_scheme
        self.percentage = perturbation_ratio
        self.cumulative_percentage = []

        if self.aug:
            prev_val = self.percentage[0]
            self.cumulative_percentage.append(self.percentage[0])
            for i in self.percentage[1:]:
                prev_val += i
                self.cumulative_percentage.append(prev_val)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        input_image, _ = self.input_image_folder.__getitem__(idx)

        if self.aug:
            # get random perturbator index by percentage
            perturbator_idx = -1
            random_var_for_perturbation = random.random()
            for per_idx, i in enumerate(self.cumulative_percentage):
                if random_var_for_perturbation < i:
                    perturbator_idx = PerturbatorEnum(per_idx)
                    break

            # augmenation of img and masks
            angle = random.randrange(-25, 25)
            trans_rand = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
            scale_rand = random.uniform(0.9, 1.1)

            # trans img
            self.affine_trans = my_transforms.Compose([my_transforms.ToPILImage(),
                                                      my_transforms.Affine(angle,
                                                                           translate=trans_rand,
                                                                           scale=scale_rand,
                                                                           fillcolor=0),
                                                      my_transforms.ToTensor(),
                                                      ])

            perturbator = None
            if perturbator_idx == PerturbatorEnum.BLACKOUT:
                perturbator = BlackoutPerturbator()
            elif perturbator_idx == PerturbatorEnum.WHITEOUT:
                perturbator = WhiteoutPerturbator()
            elif perturbator_idx == PerturbatorEnum.SMOOTHING:
                perturbator = SmoothingPerturbator()
            elif perturbator_idx == PerturbatorEnum.BINARIZATION:
                perturbator = BinarizationPerturbator()
            elif perturbator_idx == PerturbatorEnum.EDGE_DETECTION:
                perturbator = EdgeDetectionPerturbator()
            else:
                pass

            if perturbator is not None:
                input_image = perturbator.perturb(input_image)

            input_image = self.col_trans(input_image)
            input_image = self.affine_trans(input_image)

            mask = torch.empty(DataInfo.landmark_class_num, input_image.shape[1], input_image.shape[2], dtype=torch.float)
            for landmark_idx in range(0, DataInfo.landmark_class_num):
                heatmap_image, _ = self.heatmap_image_folder.__getitem__(self.data_num * landmark_idx + idx)
                mask[landmark_idx] = self.affine_trans(heatmap_image)

        else:
            mask = torch.empty(DataInfo.landmark_class_num, input_image.shape[1], input_image.shape[2], dtype=torch.float)
            for landmark_idx in range(0, DataInfo.landmark_class_num):
                heatmap_image, _ = self.heatmap_image_folder.__getitem__(self.data_num * landmark_idx + idx)
                mask[landmark_idx] = heatmap_image

        if self.dataset_type == DatasetEnum.TRAIN:
            mask = torch.pow(mask, DataInfo.pow_heatmap)

        mask = mask / mask.max()

        return [input_image, mask]
