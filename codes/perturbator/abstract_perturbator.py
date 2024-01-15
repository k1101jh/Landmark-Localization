import numpy as np
from abc import *

from data_info.data_info import DataInfo


class AbstractPerturbator(metaclass=ABCMeta):
    def __init__(self, boxing_scheme_idx, width, height):
        self.boxing_scheme_idx = boxing_scheme_idx
        self.width = width
        self.height = height

    def boxing_scheme1(self):
        rec_size_mean = 200
        rec_size_var = 10

        rec_width = np.random.normal(rec_size_mean, rec_size_var)
        rec_height = np.random.normal(rec_size_mean, rec_size_var)

        center_point = [
            np.random.uniform(rec_width / 2, self.width - (rec_width / 2)),
            np.random.uniform(rec_height / 2, self.height - (rec_height / 2)),
        ]

        # [width, height]
        left_top_point = [
            int(max(0, center_point[0] - (rec_width / 2))),
            int(max(0, center_point[1] - (rec_height / 2))),
        ]
        right_bottom_point = [
            int(min(self.width - 1, center_point[0] + (rec_width / 2))),
            int(min(self.height - 1, center_point[1] + (rec_height / 2))),
        ]

        return left_top_point, right_bottom_point

    def boxing_scheme2(self):
        min_length = 50
        max_length = 200

        center_point = [
            np.random.uniform(0, self.width),
            np.random.uniform(0, self.height),
        ]

        rec_width = np.random.uniform(min_length, max_length)
        rec_height = np.random.uniform(min_length, max_length)

        # [width, height]
        left_top_point = [
            int(max(0, center_point[0] - (rec_width / 2))),
            int(max(0, center_point[1] - (rec_height / 2))),
        ]
        right_bottom_point = [
            int(min(self.width, center_point[0] + (rec_width / 2))),
            int(min(self.height, center_point[1] + (rec_height / 2))),
        ]

        return left_top_point, right_bottom_point

    def get_rec(self):
        assert self.boxing_scheme_idx in [1, 2]
        left_top_point = None
        right_bottom_point = None

        if self.boxing_scheme_idx == 1:
            left_top_point, right_bottom_point = self.boxing_scheme1()
        elif self.boxing_scheme_idx == 2:
            left_top_point, right_bottom_point = self.boxing_scheme2()

        return left_top_point, right_bottom_point

    @abstractmethod
    def perturb(self, image):
        pass
