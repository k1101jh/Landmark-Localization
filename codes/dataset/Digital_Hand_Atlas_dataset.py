import os
import sys
import matplotlib.pyplot as plt
import random
import time
import glob
import numpy as np
import logging
from PIL import Image
from hydra.utils import instantiate

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.datasets import ImageFolder

from perturbator.binarization_perturbator import BinarizationPerturbator
from perturbator.blackout_perturbator import BlackoutPerturbator
from perturbator.edge_detection_perturbator import EdgeDetectionPerturbator
from perturbator.smoothing_perturbator import SmoothingPerturbator
from perturbator.whiteout_perturbator import WhiteoutPerturbator


class DigitalHandAtlasDataset(Dataset):
    def __init__(
        self,
        cfg,
        dataset_type,
        setup,
    ):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.setup = setup

        self.num_landmarks = cfg.num_landmarks
        self.init_trans = instantiate(cfg.init_trans)
        self.train_trans = instantiate(cfg.train_trans)

        self.input_image_paths = []
        self.heatmap_paths = []

        if dataset_type == "train":
            data_setups = [idx for idx in [1, 2, 3] if idx is not setup]
        elif dataset_type == "test":
            data_setups = [setup]
        else:
            logging.error("Wrong 'dataset_type'")
            sys.exit(0)

        for setup_dir in data_setups:
            image_dir = cfg.paths[f"setup{setup_dir}"].images
            heatmap_dir = cfg.paths[f"setup{setup_dir}"].heatmaps

            self.input_image_paths += sorted(glob.glob(image_dir + "/*"))

            for landmark_dir_path in sorted(glob.glob(heatmap_dir + "/*")):
                self.heatmap_paths.append(sorted(glob.glob(landmark_dir_path + "/*")))

            self.heatmap_paths = np.array(self.heatmap_paths).T.tolist()

        self.data_num = len(self.input_image_paths)
        self.to_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        input_image_path = self.input_image_paths[idx]
        input_image = Image.open(input_image_path)
        heatmaps = []

        for heatmap_path in self.heatmap_paths[idx]:
            heatmaps.append(self.to_tensor_transform(Image.open(heatmap_path)))

        input_image = self.image_trans(input_image)

        if self.dataset_type == "train":
            input_and_heatmap_images = input_image + heatmaps

            input_and_heatmap_images = self.data_trans(input_and_heatmap_images)

            input_image = input_and_heatmap_images[0]
            heatmaps = input_and_heatmap_images[1:]

            heatmaps = torch.pow(heatmaps, self.cfg.num_pow_heatmap)
            heatmaps = heatmaps / heatmaps.max()

        return input_image, heatmaps, input_image_path
