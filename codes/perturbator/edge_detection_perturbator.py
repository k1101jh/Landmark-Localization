from skimage import feature
import numpy as np

from perturbator.abstract_perturbator import AbstractPerturbator


class EdgeDetectionPerturbator(AbstractPerturbator):
    def __init__(self):
        self.edge_sigma_mean = 3.5
        self.edge_sigma_var = 1.5

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        crop_img = image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]]
        crop_img[:] = 1

        edge_sigma = 0
        while edge_sigma < 0:
            edge_sigma = np.random.normal(self.edge_sigma_mean, self.edge_sigma_var)

        edge_detected_img: object = feature.canny(crop_img, sigma=edge_sigma)
        image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]] = edge_detected_img

        return image
