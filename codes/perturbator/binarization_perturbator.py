from skimage.filters import threshold_otsu
import numpy as np
import torch

from perturbator.abstract_perturbator import AbstractPerturbator


class BinarizationPerturbator(AbstractPerturbator):
    def __init__(self, boxing_scheme_idx, width, height):
        super(BinarizationPerturbator, self).__init__(boxing_scheme_idx, width, height)
        self.binary_mean = 0.2
        self.binary_std = 0.15

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        cropped_image = image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]]
        cropped_image = cropped_image[0].numpy()

        threshold_value = threshold_otsu(cropped_image)

        binary_image = cropped_image > (np.random.normal(threshold_value + self.binary_mean, self.binary_std))
        binary_image = binary_image * 1
        binary_image = torch.from_numpy(binary_image)

        image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]] = binary_image

        return image
