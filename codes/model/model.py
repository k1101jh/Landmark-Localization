import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import cv2 as cv
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.io

from my_enums.dataset_enum import DatasetEnum
from network.unet import UNet
from network.attention_unet import AttentionUNet
from data_info.data_info import DataInfo
from loss import ac_loss


def gray_to_rgb(gray):
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = gray
    rgb[:, :, 1] = gray
    rgb[:, :, 2] = gray
    return rgb


class Model:
    def __init__(self, device_num: int, model_name: str, model_folder_path: str, use_tensorboard: bool):
        self.device_num = device_num
        self.device = None
        self.model_name = model_name
        self.model_path = model_folder_path
        self.model = None
        self.datasets = {}
        self.data_loaders = {}
        self.set_device()
        self.set_network()

    def set_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_num)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        print("GPU_number : ", self.device_num, "\tGPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    def generate_model_path(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        else:
            print("model exists!")

    def set_network(self):
        n_input = 1
        if DataInfo.is_attention_unet:
            self.model = AttentionUNet(n_input, DataInfo.num_landmark_class).to(self.device)
        else:
            self.model = UNet(n_input, DataInfo.num_landmark_class).to(self.device)

    def load_saved_model(self):
        model_list = os.listdir(self.model_path)
        model_list = natsorted(model_list)
        last_model_path = os.path.join(self.model_path, model_list[-1])

        loaded_model = torch.load(last_model_path, map_location=self.device)

        return loaded_model
