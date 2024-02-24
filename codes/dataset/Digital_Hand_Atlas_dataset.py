import os
import sys
import matplotlib.pyplot as plt
import random
import glob
import numpy as np
import torch
import logging
from scipy.spatial import distance
from PIL import Image
from hydra.utils import instantiate
from torch.utils.data import Dataset
from torchvision import transforms


class DigitalHandAtlasDataset(Dataset):
    def __init__(self, dataset_type, cfg, setup, **kwargs):
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.setup = setup
        self.perturbation_ratio = self.cfg.train.perturbation_ratio

        self.num_landmarks = cfg.dataset.num_landmarks

        self.input_trans = transforms.Compose(instantiate(self.cfg.train.input_trans))
        self.colorjitter_trans = transforms.Compose(instantiate(self.cfg.train.colorjitter_trans))
        self.affine_trans = transforms.Compose(instantiate(self.cfg.train.affine_trans))

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
            heatmap_paths = []
            image_dir = cfg.dataset.paths[f"setup{setup_dir}"].images
            heatmap_dir = cfg.dataset.paths[f"setup{setup_dir}"].heatmaps

            self.input_image_paths += sorted(glob.glob(image_dir + "/*"))

            for landmark_dir_path in sorted(glob.glob(heatmap_dir + "/*")):
                heatmap_paths.append(sorted(glob.glob(landmark_dir_path + "/*")))

            self.heatmap_paths += np.array(heatmap_paths).T.tolist()

        self.data_num = len(self.input_image_paths)

        # Perturbator
        self.perturbators = instantiate(cfg.train.perturbators)
        self.cumulative_percentage = []

        if self.dataset_type == "train":
            prev_val = self.perturbation_ratio[0]
            self.cumulative_percentage.append(self.perturbation_ratio[0])
            for i in self.perturbation_ratio[1:]:
                prev_val += i
                self.cumulative_percentage.append(prev_val)

        self.to_pil_transform = transforms.ToPILImage()
        self.to_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return self.data_num

    def calc_dists(self, landmark_points, gt_points):
        """
        손목 부위 랜드마크( , ) 사이의 거리를 50mm로 가정
        """
        dists = []
        wrist_length = distance.euclidean(gt_points[0], gt_points[4])

        for landmark_point, gt_point in zip(landmark_points, gt_points):
            dist = distance.euclidean(landmark_point, gt_point)
            dist *= 50 / wrist_length
            dists.append(dist)
        return dists

    def __getitem__(self, idx):
        input_image_path = self.input_image_paths[idx]
        input_image = Image.open(input_image_path)
        heatmaps = []
        meta_data = {}
        meta_data["image_path"] = input_image_path
        meta_data["original_size"] = input_image.size

        for heatmap_path in self.heatmap_paths[idx]:
            heatmaps.append(self.to_tensor_transform(Image.open(heatmap_path)))

        heatmaps = torch.stack(heatmaps, dim=0)
        heatmaps = heatmaps.squeeze(1)
        input_image = self.input_trans(input_image)

        if self.dataset_type == "train":
            input_image = self.colorjitter_trans(input_image)
            input_image = self.to_tensor_transform(input_image)

            input_and_heatmap_images = torch.cat((input_image, heatmaps), dim=0)
            input_and_heatmap_images = self.affine_trans(input_and_heatmap_images)

            input_image = input_and_heatmap_images[0, None]
            heatmaps = input_and_heatmap_images[1:]

            # get random perturbator index by percentage
            perturbator_idx = -1
            random_var_for_perturbation = random.random()
            for per_idx, i in enumerate(self.cumulative_percentage):
                if random_var_for_perturbation < i:
                    perturbator_idx = per_idx
                    break

            if perturbator_idx != -1:
                perturbator = self.perturbators[perturbator_idx]
                input_image = perturbator.perturb(input_image)

            heatmaps = torch.pow(heatmaps, self.cfg.train.num_pow_heatmap)
            heatmaps = heatmaps / heatmaps.max()

        else:
            input_image = self.to_tensor_transform(input_image)

        return input_image, heatmaps, meta_data
