import numpy as np
from abc import *

from data_info.data_info import DataInfo


class AbstractPerturbator(metaclass=ABCMeta):
    @staticmethod
    def boxing_scheme1():
        rec_size_mean = 200
        rec_size_var = 10

        width = np.random.normal(rec_size_mean, rec_size_var)
        height = np.random.normal(rec_size_mean, rec_size_var)

        center_point = [np.random.uniform(width / 2, DataInfo.resized_image_size[0] - (width / 2)),
                        np.random.uniform(height / 2, DataInfo.resized_image_size[1] - (height / 2))]

        # [width, height]
        left_top_point = [int(max(0, center_point[0] - (width / 2))),
                          int(max(0, center_point[1] - (height / 2)))]
        right_bottom_point = [int(min(DataInfo.resized_image_size[0] - 1, center_point[0] + (width / 2))),
                              int(min(DataInfo.resized_image_size[1] - 1, center_point[1] + (height / 2)))]

        return left_top_point, right_bottom_point

    @staticmethod
    def boxing_scheme2():
        min_length = 50
        max_length = 200

        center_point = [np.random.uniform(0, DataInfo.resized_image_size[0]),
                        np.random.uniform(0, DataInfo.resized_image_size[1])]

        width = np.random.uniform(min_length, max_length)
        height = np.random.uniform(min_length, max_length)

        # [width, height]
        left_top_point = [int(max(0, center_point[0] - (width / 2))),
                          int(max(0, center_point[1] - (height / 2)))]
        right_bottom_point = [int(min(DataInfo.resized_image_size[0], center_point[0] + (width / 2))),
                              int(min(DataInfo.resized_image_size[1], center_point[1] + (height / 2)))]

        return left_top_point, right_bottom_point

    def get_rec(self):
        left_top_point = None
        right_bottom_point = None
        while True:
            try:
                if DataInfo.boxing_scheme == 1:
                    left_top_point, right_bottom_point = self.boxing_scheme1()
                elif DataInfo.boxing_scheme == 2:
                    left_top_point, right_bottom_point = self.boxing_scheme2
                break
            except ValueError:
                continue
        return left_top_point, right_bottom_point

    @abstractmethod
    def perturb(self, image):
        pass
