import torch
import torch.nn as nn
import numpy as np

from data_info.data_info import DataInfo


class ACloss(nn.Module):
    def __init__(self):
        super(ACloss, self).__init__()
        self.height = DataInfo.resized_image_size[1]
        self.width = DataInfo.resized_image_size[0]
        self.landmark_class_num = DataInfo.landmark_class_num

    def get_angle_matrix(self, inp_mat):
        np_array1 = inp_mat - [self.height / 2, self.width / 2]
        np_array2 = inp_mat - [self.height / 2, self.width / 2]

        arccos_val = np.dot(np_array1, np_array2.transpose()) / (
                    np.linalg.norm(np_array1, axis=1).reshape(np_array1.shape[0], 1) *
                    np.linalg.norm(np_array2, axis=1))

        arccos_val = np.where(arccos_val < -1, -1, arccos_val)
        arccos_val = np.where(arccos_val > 1, 1, arccos_val)

        angle_matrix = np.arccos(arccos_val)

        angle_matrix[np.isnan(angle_matrix)] = 0

        return angle_matrix

    @staticmethod
    def get_dist_matrix(inp_mat):
        y_meshgrid1, y_meshgrid2 = np.meshgrid(inp_mat[:, 0], inp_mat[:, 0])
        x_meshgrid1, x_meshgrid2 = np.meshgrid(inp_mat[:, 1], inp_mat[:, 1])
        dist = np.sqrt((y_meshgrid1 - y_meshgrid2) ** 2 + (x_meshgrid1 - x_meshgrid2) ** 2)

        return dist

    def get_angle_and_dist_loss(self, output, target):  # tensor(1,landmark_num,h,w)
        angle_loss = 0.0
        dist_loss = 0.0
        for batch in range(target.size(0)):
            output_matrix = np.zeros((self.landmark_class_num, 2))
            target_matrix = np.zeros((self.landmark_class_num, 2))
            for landmark_num in range(0, self.landmark_class_num):
                output_image = output[batch][landmark_num]
                output_image = output_image.cpu()
                target_image = target[batch][landmark_num]
                target_image = target_image.cpu()
                output_max_point_np_array = np.array(np.where(output_image == output_image.max()))
                target_max_point_np_array = np.array(np.where(target_image == target_image.max()))
                output_matrix[landmark_num] = output_max_point_np_array[:, 0]
                target_matrix[landmark_num] = target_max_point_np_array[:, 0]

            output_angle = self.get_angle_matrix(output_matrix)
            target_angle = self.get_angle_matrix(target_matrix)
            angle_loss += np.mean(np.abs(output_angle - target_angle))

            output_dist = self.get_dist_matrix(output_matrix)
            target_dist = self.get_dist_matrix(target_matrix)
            dist_loss += np.mean(np.abs(output_dist - target_dist))

        return angle_loss, dist_loss

    def forward(self, output, target):
        l2_loss = torch.mean(torch.pow((output - target), 2))

        angle_loss, dist_loss = self.get_angle_and_dist_loss(output, target)
        w_loss = (1 + angle_loss) + np.log(dist_loss + 1e-10)
        loss = torch.mul(l2_loss, w_loss)

        return loss, l2_loss, w_loss, angle_loss, dist_loss
