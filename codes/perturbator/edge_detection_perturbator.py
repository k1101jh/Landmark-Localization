import numpy as np
import torch
from skimage import feature

from perturbator.abstract_perturbator import AbstractPerturbator


class EdgeDetectionPerturbator(AbstractPerturbator):
    def __init__(self, boxing_scheme_idx, width, height):
        super(EdgeDetectionPerturbator, self).__init__(boxing_scheme_idx, width, height)
        self.edge_sigma_mean = 3.5
        self.edge_sigma_var = 1.5

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        cropped_image = image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]]
        cropped_image = cropped_image[0].numpy()

        edge_sigma = 0
        while edge_sigma < 0:
            edge_sigma = np.random.normal(self.edge_sigma_mean, self.edge_sigma_var)

        edge_detected_image: object = feature.canny(cropped_image, sigma=edge_sigma)
        edge_detected_image = torch.from_numpy(edge_detected_image)

        image[
            :, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]
        ] = edge_detected_image

        return image
