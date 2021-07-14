import os
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
import time
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import cv2 as cv
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.io
from tensorboardX import SummaryWriter

from dataset.dataset import MyDataset
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
        self.writer = None
        self.use_tensorboard = use_tensorboard

        self.set_device()
        self.set_network()

        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(DataInfo.tensorboard_path, self.model_name))

    def set_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_num)
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        print("GPU_number : ", self.device_num, '\tGPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))

    def generate_model_path(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        else:
            print("model exists!")

    def set_network(self):
        n_input = 1
        if DataInfo.is_attention_unet:
            self.model = AttentionUNet(n_input, DataInfo.landmark_class_num).to(self.device)
        else:
            self.model = UNet(n_input, DataInfo.landmark_class_num).to(self.device)

    def load_saved_model(self):
        model_list = os.listdir(self.model_path)
        model_list = natsorted(model_list)
        last_model_path = os.path.join(self.model_path, model_list[-1])

        loaded_model = torch.load(last_model_path, map_location=self.device)

        return loaded_model

    def train(self, load: bool, perturbation_percentage: list, num_epochs: int, lr=1e-4):
        self.generate_model_path()
        loaded_model = None

        if load:
            loaded_model = self.load_saved_model()
            perturbation_percentage = loaded_model['perturbation_percentage']

        # generate dataset
        self.datasets = {
            'train': MyDataset(dataset_type=DatasetEnum.TRAIN, perturbation_ratio=perturbation_percentage, aug=True),
            'test': MyDataset(dataset_type=DatasetEnum.TEST1, perturbation_ratio=None, aug=False)
        }

        self.data_loaders = {
            'train': DataLoader(self.datasets['train'], batch_size=DataInfo.batch_size, shuffle=True, num_workers=4),
            'test': DataLoader(self.datasets['test'], batch_size=DataInfo.batch_size, shuffle=False, num_workers=4)
        }

        base_file_name = 'E_{}_best_loss_{:0.8f}_w_loss_{:0.6f}'

        best_loss = 10
        test_interval = 10
        epoch_loss = 0.0
        epoch_w_loss = 0.0
        epoch_time = 0
        start_epoch = 0

        loss_function = ac_loss.ACloss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0, -1)

        if load:
            self.model.load_saved_model(loaded_model['model'])
            start_epoch = loaded_model['epoch']
            optimizer.load_state_dict(loaded_model['optim'])
            scheduler.load_state_dict(loaded_model['scheduler'])

        tqdm_epoch = tqdm(range(start_epoch, num_epochs), initial=start_epoch, total=num_epochs,
                          ncols=100, desc='loss: {:0.8f}'.format(epoch_loss) +
                                          ' w_loss: {:0.8f}'.format(epoch_w_loss) +
                                          ' epoch_time: {:0.2f}'.format(epoch_time))

        for epoch in tqdm_epoch:
            epoch = epoch + 1
            epoch_start_time = time.time()

            if epoch % test_interval == 0:
                uu = ['train', 'test']
            else:
                uu = ['train']

            for phase in uu:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in self.data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)

                        loss, l2_loss, w_loss, angle_loss, dist_loss = loss_function.forward(outputs, labels)
                        metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
                        metrics['w_loss'] += w_loss * labels.size(0)
                        metrics['angle_loss'] += angle_loss * labels.size(0)
                        metrics['dist_loss'] += dist_loss * labels.size(0)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                    if self.use_tensorboard and epoch == 1:
                        img_grid = torchvision.utils.make_grid(inputs)
                        self.writer.add_image('sample augmentations', img_grid, epoch_samples)
                        self.writer.close()

                if phase == 'train':
                    scheduler.step()

                # print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples
                epoch_w_loss = metrics['w_loss'] / epoch_samples
                epoch_angle_loss = metrics['angle_loss'] / epoch_samples
                epoch_dist_loss = metrics['dist_loss'] / epoch_samples

                if self.use_tensorboard:
                    self.writer.add_scalar(phase + ' loss', epoch_loss, epoch)
                    self.writer.add_scalar(phase + ' w loss', epoch_w_loss, epoch)
                    self.writer.add_scalar(phase + ' angle loss', epoch_angle_loss, epoch)
                    self.writer.add_scalar(phase + ' dist loss', epoch_dist_loss, epoch)
                    if epoch % 10 == 0:
                        self.writer.close()

                if phase == 'test' and ((epoch in DataInfo.save_epoch) or epoch_loss < best_loss):
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        file_name = base_file_name + '_best.pth'
                    else:
                        file_name = base_file_name + '.pth'

                    state = {
                        'perturbation_percentage': perturbation_percentage,
                        'model': self.model.state_dict(),
                        'epoch': epoch,
                        'optim': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }

                    save_file_path = os.path.join(self.model_path,
                                                  file_name.format(epoch, best_loss, epoch_w_loss))
                    torch.save(state, save_file_path)

            epoch_time = time.time() - epoch_start_time
            tqdm_epoch.set_description(desc='loss: {:0.8f}'.format(epoch_loss) +
                                            ' w_loss: {:0.8f}'.format(epoch_w_loss) +
                                            ' epoch_time: {:0.2f}'.format(epoch_time))

    def test(self, model_path, save_result_images=False, dataset_type=DatasetEnum.TEST1):
        # generate result image save path
        if save_result_images:
            os.makedirs(DataInfo.output_image_save_path, exist_ok=True)
            os.makedirs(DataInfo.gt_image_save_path, exist_ok=True)
            os.makedirs(DataInfo.output_heatmap_image_save_path, exist_ok=True)
            os.makedirs(DataInfo.gt_heatmap_image_save_path, exist_ok=True)

            for image_idx in range(0, 400):
                os.makedirs(os.path.join(DataInfo.output_heatmap_image_save_path, str(image_idx + 1)), exist_ok=True)
                os.makedirs(os.path.join(DataInfo.gt_heatmap_image_save_path, str(image_idx + 1)), exist_ok=True)

        test_start_time = time.time()

        # generate dataset
        test_dataset = MyDataset(dataset_type=dataset_type, aug=False)
        test_dataloader = DataLoader(test_dataset, batch_size=DataInfo.batch_size, shuffle=False, num_workers=4)

        # load_model
        loaded_model = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(loaded_model['model'])
        self.model.eval()

        # load gt numpy
        # [image_num, width, height]
        gt_numpy = np.load(DataInfo.landmark_gt_numpy_path, allow_pickle=True)
        image_start_idx = 0
        if dataset_type == DatasetEnum.TEST1:
            image_start_idx = 150
        elif dataset_type == DatasetEnum.TEST2:
            image_start_idx = 300
        elif dataset_type == DatasetEnum.TRAIN:
            image_start_idx = 0

        distance_list = []
        landmark_point_list = []
        result_img = None
        gt_img = None

        count = 0

        tqdm_dataloader = tqdm(test_dataloader, ncols=130, desc='accuracy: {:0.4f}'.format(0))

        for data_index, data in enumerate(tqdm_dataloader):
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            outputs = outputs.data

            for batch in range(inputs.size(0)):
                image_index = image_start_idx + DataInfo.batch_size * data_index + batch
                input_image = inputs[batch]
                label = labels[batch]
                output = outputs[batch]

                # generate image to draw landmark on it
                if save_result_images or self.use_tensorboard:
                    result_img = gray_to_rgb(input_image[0].cpu())
                    result_img *= 255
                    result_img = cv.resize(result_img, tuple(DataInfo.original_image_size),
                                           interpolation=cv.INTER_AREA)

                    gt_img = gray_to_rgb(input_image[0].cpu())
                    gt_img *= 255
                    gt_img = cv.resize(gt_img, tuple(DataInfo.original_image_size),
                                       interpolation=cv.INTER_AREA)

                for landmark_count in range(0, DataInfo.landmark_class_num):
                    landmark_heatmap = output[landmark_count]
                    landmark_heatmap = landmark_heatmap.cpu()
                    landmark_heatmap_gt = label[landmark_count]
                    landmark_heatmap_gt = landmark_heatmap_gt.cpu()

                    # [height, width]
                    landmark_heatmap_max = np.array(
                        np.where(landmark_heatmap > landmark_heatmap.max() * DataInfo.heatmap_remain_ratio))
                    landmark_heatmap_max = landmark_heatmap_max.mean(axis=1)

                    # [width, height]
                    gt_point = gt_numpy[image_index][landmark_count]

                    # landmark_heatmap_max: [height, width]
                    # original_image_size: [width, height]
                    original_size_heatmap_max_point_x = landmark_heatmap_max[1] * (
                                DataInfo.original_image_size[0] / DataInfo.resized_image_size[0])
                    original_size_heatmap_max_point_y = landmark_heatmap_max[0] * (
                                DataInfo.original_image_size[1] / DataInfo.resized_image_size[1])
                    original_size_heatmap_max_point = [original_size_heatmap_max_point_x,
                                                       original_size_heatmap_max_point_y]

                    landmark_point_list.append(original_size_heatmap_max_point)

                    dist = distance.euclidean(original_size_heatmap_max_point, gt_point)

                    distance_list.append(dist)

                    if save_result_images or self.use_tensorboard:
                        landmark_point = (
                            int(original_size_heatmap_max_point_x), int(original_size_heatmap_max_point_y))
                        cv.circle(result_img, landmark_point, 17, (255, 0, 0), -1)
                        # cv.putText(result_img, str(landmark_count),
                        #            (int(landmark_point[0]) - 50, int(landmark_point[1]) - 10),
                        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        cv.circle(gt_img, (int(gt_point[0]), int(gt_point[1])), 17, (0, 0, 255), -1)
                        # cv.putText(result_img2, str(landmark_count),
                        #            (int(gt_point[0]) - 50, int(gt_point[1]) - 10),
                        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        landmark_heatmap_gt = torch.pow(landmark_heatmap_gt, 7)
                        landmark_heatmap_gt /= landmark_heatmap_gt.max()
                        plt.imsave(os.path.join(DataInfo.output_heatmap_image_save_path, str(image_index + 1),
                                                str(landmark_count + 1) + ".png"),
                                   landmark_heatmap, cmap='gray')
                        plt.imsave(os.path.join(DataInfo.gt_heatmap_image_save_path, str(image_index + 1),
                                                str(landmark_count + 1) + ".png"),
                                   landmark_heatmap_gt, cmap='gray')

                    if dist < 20:
                        count = count + 1

                if save_result_images:
                    cv.imwrite(os.path.join(DataInfo.output_image_save_path, str(image_index + 1) + '.png'), result_img)
                    cv.imwrite(os.path.join(DataInfo.gt_image_save_path, str(image_index + 1) + '.png'), gt_img)

                if self.use_tensorboard:
                    fig = plt.figure(figsize=(4, 5))
                    fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
                    plt.imshow(gt_img.astype('uint8'))
                    fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
                    plt.imshow(result_img.astype('uint8'))

                    self.writer.add_figure('preds vs result', fig, image_index)
                    plt.close(fig)

            tqdm_dataloader.set_description(
                desc='accuracy: {:0.4f}'.format(count /
                                                ((DataInfo.batch_size * data_index) * DataInfo.landmark_class_num)))

        if self.use_tensorboard:
            self.writer.close()

        # calculate accuracy
        mm_per_pixel = 10
        distance_numpy = np.array(distance_list)

        # calculate accuracy of all landmarks
        acc_2mm = np.mean(distance_numpy < mm_per_pixel * 2)
        acc_2_5mm = np.mean(distance_numpy < mm_per_pixel * 2.5)
        acc_3mm = np.mean(distance_numpy < mm_per_pixel * 3)
        acc_4mm = np.mean(distance_numpy < mm_per_pixel * 4)

        print("2mm: %.4f\t" % acc_2mm,
              "2.5mm: %.4f\t" % acc_2_5mm,
              "3mm: %.4f\t" % acc_3mm,
              "4mm: %.4f\t" % acc_4mm,
              "ave: %.4f\t" % np.mean(distance_numpy),
              "median: %.4f\t" % np.median(distance_numpy),
              "std: %.4f\t" % np.std(distance_numpy))

        # calculate accuracy by landmarks
        distance_numpy = np.array(distance_list).reshape(-1, DataInfo.landmark_class_num)
        landmark_point_numpy = np.asarray(landmark_point_list).reshape(-1, DataInfo.landmark_class_num)
        accuracy_numpy = np.zeros((DataInfo.landmark_class_num, 4))

        for i in range(DataInfo.landmark_class_num):
            accuracy_numpy[i][0] = np.mean(distance_numpy[:, i] < mm_per_pixel * 2)
            accuracy_numpy[i][1] = np.mean(distance_numpy[:, i] < mm_per_pixel * 2.5)
            accuracy_numpy[i][2] = np.mean(distance_numpy[:, i] < mm_per_pixel * 3)
            accuracy_numpy[i][3] = np.mean(distance_numpy[:, i] < mm_per_pixel * 4)

        for i in range(DataInfo.landmark_class_num):
            print("point num :%2d " % (i + 1), end=' ')
            for j in range(4):
                print('%.4f\t' % (accuracy_numpy[i][j]), end=' ')

            point_mean = np.mean(distance_numpy[:, i])
            point_std = np.std(distance_numpy[:, i])
            print('%.4f\t' % point_mean, end=' ')
            print('%.4f\t' % point_std)

        print("test_time:", time.time() - test_start_time)

        # save test result
        # save test result as mat file
        distance_mat = {'distance': distance_numpy.tolist()}
        distance_mat_file_path = os.path.dirname(model_path) + '/distance.mat'
        scipy.io.savemat(distance_mat_file_path, distance_mat)

        # save test result as numpy file
        test_accuracy_numpy_file_path = os.path.dirname(model_path) + '/test_result.npy'
        landmark_point_numpy_file_path = os.path.dirname(model_path) + '/pred_result.npy'

        np.save(test_accuracy_numpy_file_path, accuracy_numpy)
        np.save(landmark_point_numpy_file_path, landmark_point_numpy)

        # save test result as txt file
        test_result_file_name = os.path.dirname(model_path) + '/test_result.txt'

        f = open(test_result_file_name, 'w')
        f.write("2mm: %.4f\t" % acc_2mm +
                "2.5mm: %.4f\t" % acc_2_5mm +
                "3mm: %.4f\t" % acc_3mm +
                "4mm: %.4f\t" % acc_4mm +
                "ave: %.4f\t" % np.mean(distance_numpy) +
                "median: %.4f\t" % np.median(distance_numpy) +
                "std: %.4f" % np.std(distance_numpy) + '\n')

        f.write('\n')
        f.write('\t\t2mm\t2.5mm\t3mm\t4mm\tmean\tstd\n')
        for i in range(DataInfo.landmark_class_num):
            f.write("point num :%2d " % (i + 1) + '\t')
            for j in range(4):
                f.write('%.4f\t' % (accuracy_numpy[i][j]))

            point_mean = np.mean(distance_numpy[:, i])
            point_std = np.std(distance_numpy[:, i])
            f.write('%.4f \t' % point_mean)
            f.write('%.4f\t' % point_std + '\n')

        f.close()
