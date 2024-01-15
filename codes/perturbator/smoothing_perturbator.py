import numpy as np
import torch
from skimage.transform import resize

from perturbator.abstract_perturbator import AbstractPerturbator


class SmoothingPerturbator(AbstractPerturbator):
    def __init__(self, boxing_scheme_idx, width, height):
        super(SmoothingPerturbator, self).__init__(boxing_scheme_idx, width, height)
        self.smoothing_mean = 0.2
        self.smoothing_var = 0.15

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        cropped_image = image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]]
        cropped_image = cropped_image[0].numpy()

        smoothing_ratio = np.random.normal(self.smoothing_mean, self.smoothing_var)

        crop_size = cropped_image.shape
        resize_image_size = [abs(round(crop_size[0] * smoothing_ratio)), abs(round(crop_size[1] * smoothing_ratio))]
        if resize_image_size[0] <= 5:
            resize_image_size[0] = 5
        if resize_image_size[1] <= 5:
            resize_image_size[1] = 5

        smoothing_image = resize(cropped_image, resize_image_size)
        smoothing_image = resize(smoothing_image, crop_size)
        smoothing_image = torch.from_numpy(smoothing_image)

        image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]] = smoothing_image

        return image
