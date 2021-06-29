from skimage.filters import threshold_otsu
import numpy as np
import torch

from perturbator.abstract_perturbator import AbstractPerturbator


class BinarizationPerturbator(AbstractPerturbator):
    def __init__(self):
        self.binary_mean = 0.2
        self.binary_std = 0.15

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        crop_img = image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]]

        crop_img = crop_img[0].numpy()

        threshold_value = threshold_otsu(crop_img)

        binary = crop_img > (np.random.normal(threshold_value + self.binary_mean, self.binary_std))
        binary = binary * 1
        binary = torch.from_numpy(binary)

        image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]] = binary

        return image
