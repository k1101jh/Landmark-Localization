import numpy as np
import torch
from skimage.transform import resize

from perturbator.abstract_perturbator import AbstractPerturbator


class SmoothingPerturbator(AbstractPerturbator):
    def __init__(self):
        self.smoothing_mean = 0.2
        self.smoothing_var = 0.15

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        crop_img = image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]]
        crop_img[:] = 1

        smoothing_ratio = np.random.normal(self.smoothing_mean, self.smoothing_var)

        crop_size = crop_img.shape
        resize_image_size = [abs(round(crop_size[0] * smoothing_ratio)), abs(round(crop_size[1] * smoothing_ratio))]
        if resize_image_size[0] <= 5:
            resize_image_size[0] = 5
        if resize_image_size[1] <= 5:
            resize_image_size[1] = 5

        smoothing_img = resize(crop_img, resize_image_size)
        smoothing_img = resize(smoothing_img, crop_size)

        image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]] = torch.tensor(smoothing_img)

        return image
