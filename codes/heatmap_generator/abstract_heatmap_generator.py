import numpy as np
from abc import *


class AbstractHeatmapGenerator(metaclass=ABCMeta):
    def __init__(self, width, height, scale):
        self.width = width
        self.height = height
        self.scale = scale

    def generate_dist_matrix(self, landmark_point, resized=True, original_width=None, original_height=None):
        if resized:
            resized_landmark_point = landmark_point
        else:
            assert original_width != None and original_height != None
            resized_landmark_point = [
                landmark_point[0] * (self.width / original_width),
                landmark_point[1] * (self.height / original_height),
            ]

        x_linespace_mtx = np.arange(-resized_landmark_point[0], self.width - resized_landmark_point[0])
        y_linespace_mtx = np.arange(-resized_landmark_point[1], self.height - resized_landmark_point[1])

        x_axis_mtx, y_axis_mtx = np.meshgrid(x_linespace_mtx, y_linespace_mtx)

        return x_axis_mtx, y_axis_mtx

    @abstractmethod
    def get_heatmap_image(self, landmark_point, resized, original_width, original_height):
        pass
