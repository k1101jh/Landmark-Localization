import numpy as np
from PIL import Image

from heatmap_generator.abstract_heatmap_generator import AbstractHeatmapGenerator


class AnisotropicLaplaceHeatmapGenerator(AbstractHeatmapGenerator):
    def __init__(self):
        super().__init__()

    def get_heatmap_image(self, landmark_point):
        x_axis_mtx, y_axis_mtx = self.generate_dist_matrix(landmark_point)

        b = 15

        laplace_img = 1 / (2 * b) * np.exp(-1 * np.abs(np.sqrt(x_axis_mtx ** 2 + y_axis_mtx ** 2)) / (b ** 2))
        laplace_img = laplace_img / laplace_img.max()
        laplace_img *= 255
        laplace_img = Image.fromarray(laplace_img)
        laplace_img = laplace_img.convert('RGB')

        return laplace_img
