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
from enums.dataset_enum import DatasetEnum
from enums.perturbator_enum import PerturbatorEnum


class MyDataset(Dataset):
    def __init__(self, dataset_type=DatasetEnum.TEST1, perturbation_ratio=None, aug=True):

        init_trans = transforms.Compose([transforms.Resize((DataInfo.resized_image_size[1], DataInfo.resized_image_size[0])),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])

        self.dataset_type = dataset_type
        self.datainfo = torchvision.datasets.ImageFolder(
            root=os.path.join(DataInfo.gp_path, 'train_test_data', self.dataset_type.name.lower()), transform=init_trans)
        self.mask_num = len(self.datainfo.classes) - 1
        self.data_num = int(len(self.datainfo) / len(self.datainfo.classes))
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
        image, _ = self.datainfo.__getitem__(idx)

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

            # trans img with masks
            self.input_trans = my_transforms.Compose([my_transforms.ToPILImage(),
                                                      my_transforms.Affine(angle,
                                                                           translate=trans_rand,
                                                                           scale=scale_rand,
                                                                           fillcolor=0),
                                                      my_transforms.ToTensor(),
                                                      ])

            self.mask_trans = my_transforms.Compose([my_transforms.ToPILImage(),
                                                     my_transforms.Affine(angle,
                                                                          translate=trans_rand,
                                                                          scale=scale_rand,
                                                                          fillcolor=0),
                                                     my_transforms.ToTensor(),
                                                     ])

            self.col_trans = my_transforms.Compose([my_transforms.ToPILImage(),
                                                    my_transforms.ColorJitter(brightness=random.random(),
                                                                              contrast=random.random(),
                                                                              saturation=random.random(),
                                                                              hue=random.random() / 2
                                                                              ),
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
                image = perturbator.perturb(image)

            image = self.col_trans(image)
            image = self.input_trans(image)

            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                m, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = self.mask_trans(m)

        else:
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                m, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = m

        if self.dataset_type == DatasetEnum.TRAIN:
            mask = torch.pow(mask, DataInfo.pow_heatmap)

        mask = mask / mask.max()

        return [image, mask]
