from perturbator.abstract_perturbator import AbstractPerturbator


class BlackoutPerturbator(AbstractPerturbator):
    def __init__(self, boxing_scheme_idx, width, height):
        super(BlackoutPerturbator, self).__init__(boxing_scheme_idx, width, height)
        self.binary_mean = 0.2
        self.binary_std = 0.15

    def perturb(self, image):
        left_top_point, right_bottom_point = self.get_rec()

        crop_img = image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]]
        crop_img[:] = 0
        # crop_img = crop_img[0].numpy()
        #
        # crop_img[:] = 0
        # crop_img = torch.from_numpy(crop_img)

        image[:, left_top_point[1] : right_bottom_point[1], left_top_point[0] : right_bottom_point[0]] = crop_img

        return image
