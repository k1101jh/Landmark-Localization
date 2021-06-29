import numpy as np
from abc import *

from data_info.data_info import DataInfo


class AbstractHeatmapGenerator(metaclass=ABCMeta):
    def __init__(self):
        self.width, self.height = DataInfo.resized_image_size

    def resize_landmark_point(self, landmark_point):
        resized_landmark_point = [landmark_point[0] * (self.width / DataInfo.original_image_size[0]),
                                  landmark_point[1] * (self.height / DataInfo.original_image_size[1])]

        return resized_landmark_point

    def generate_dist_matrix(self, landmark_point):
        resized_landmark_point = self.resize_landmark_point(landmark_point)

        x_linespace_mtx = np.arange(-resized_landmark_point[0], self.width - resized_landmark_point[0])
        y_linespace_mtx = np.arange(-resized_landmark_point[1], self.height - resized_landmark_point[1])

        x_axis_mtx, y_axis_mtx = np.meshgrid(x_linespace_mtx, y_linespace_mtx)

        return x_axis_mtx, y_axis_mtx

    @abstractmethod
    def get_heatmap_image(self, landmark_point):
        pass
