import torch
import torch.nn as nn
import numpy as np

from data_info.data_info import DataInfo


class ACloss(nn.Module):
    def __init__(self, width, height, num_landmarks):
        super(ACloss, self).__init__()
        self.width = width
        self.height = height
        self.num_landmarks = num_landmarks

    def get_angle_matrix(self, inp_mat):
        center_pos_tensor = torch.tensor([self.height / 2, self.width / 2]).cuda()
        sub_tensor1 = torch.sub(inp_mat, center_pos_tensor)
        sub_tensor2 = torch.sub(inp_mat, center_pos_tensor)

        arccos_val = torch.matmul(sub_tensor1, torch.transpose(sub_tensor2, 0, 1)) / (
            torch.linalg.norm(sub_tensor1, axis=1).reshape(sub_tensor1.shape[0], 1)
            * torch.linalg.norm(sub_tensor2, axis=1)
        )

        arccos_val = torch.where(arccos_val < -1, -1, arccos_val)
        arccos_val = torch.where(arccos_val > 1, 1, arccos_val)

        angle_matrix = torch.arccos(arccos_val)

        angle_matrix[torch.isnan(angle_matrix)] = 0

        del center_pos_tensor
        del sub_tensor1
        del sub_tensor2

        return angle_matrix

    @staticmethod
    def get_dist_matrix(inp_mat):
        y_meshgrid1, y_meshgrid2 = torch.meshgrid(inp_mat[:, 0], inp_mat[:, 0], indexing="ij")
        x_meshgrid1, x_meshgrid2 = torch.meshgrid(inp_mat[:, 1], inp_mat[:, 1], indexing="ij")
        dist = torch.sqrt((y_meshgrid1 - y_meshgrid2) ** 2 + (x_meshgrid1 - x_meshgrid2) ** 2)

        return dist

    def get_angle_and_dist_loss(self, output, target):  # tensor(1,landmark_num,h,w)
        angle_loss = 0.0
        dist_loss = 0.0
        for image_idx in range(target.size(0)):
            output_matrix = torch.zeros((self.num_landmarks, 2)).cuda()
            target_matrix = torch.zeros((self.num_landmarks, 2)).cuda()
            for landmark_num in range(0, self.num_landmarks):
                output_image = output[image_idx][landmark_num]
                # output_image = output_image.cpu()
                target_image = target[image_idx][landmark_num]
                # target_image = target_image.cpu()
                output_matrix[landmark_num] = (output_image == torch.max(output_image)).nonzero()[0]
                target_matrix[landmark_num] = (target_image == torch.max(target_image)).nonzero()[0]
                # output_max_point_np_array = torch.where(output_image == output_image.max())
                # target_max_point_np_array = torch.where(target_image == target_image.max())
                # output_matrix[landmark_num] = output_max_point_np_array[:, 0]
                # target_matrix[landmark_num] = target_max_point_np_array[:, 0]

            output_angle = self.get_angle_matrix(output_matrix)
            target_angle = self.get_angle_matrix(target_matrix)
            angle_loss += torch.mean(torch.abs(output_angle - target_angle))

            output_dist = self.get_dist_matrix(output_matrix)
            target_dist = self.get_dist_matrix(target_matrix)
            dist_loss += torch.mean(torch.abs(output_dist - target_dist))

            del output_matrix
            del target_matrix

        return angle_loss, dist_loss

    def forward(self, output, target):
        l2_loss = torch.mean(torch.pow((output - target), 2))

        angle_loss, dist_loss = self.get_angle_and_dist_loss(output, target)
        w_loss = (1 + angle_loss) + torch.log(dist_loss + 1e-10)
        loss = torch.mul(l2_loss, w_loss)

        return loss, l2_loss, w_loss, angle_loss, dist_loss
