import numpy as np
from PIL import Image

from heatmap_generator.abstract_heatmap_generator import AbstractHeatmapGenerator


class AnisotropicLaplaceHeatmapGenerator(AbstractHeatmapGenerator):
    def get_heatmap_image(self, landmark_point, resized=True, original_width=None, original_height=None):
        x_axis_mtx, y_axis_mtx = self.generate_dist_matrix(
            landmark_point, resized=resized, original_width=original_width, original_height=original_height
        )

        laplace_img = (1 / (2 * self.scale)) * np.exp(-((abs(x_axis_mtx) + abs(y_axis_mtx)) / (self.scale**2)))
        laplace_img = laplace_img / laplace_img.max()
        laplace_img *= 255
        laplace_img = Image.fromarray(laplace_img)
        laplace_img = laplace_img.convert("L")

        return laplace_img
