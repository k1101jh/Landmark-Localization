
from perturbator.abstract_perturbator import AbstractPerturbator


class WhiteoutPerturbator(AbstractPerturbator):
    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        crop_img = image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]]
        crop_img[:] = 1

        image[:, left_top_point[1]:right_bottom_point[1], left_top_point[0]:right_bottom_point[0]] = crop_img

        return image
