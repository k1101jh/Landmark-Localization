import os
import numpy as np
from tqdm import tqdm
import copy
import shutil

from data_info.data_info import DataInfo
from heatmap_generator.anisotropic_laplace_heatmap_generator import AnisotropicLaplaceHeatmapGenerator


class DatasetGenerator:
    @classmethod
    def generate_dataset(cls):
        print('\nStep 1: Generate gt numpy file\n')
        cls.generate_gt_numpy()

        print('\nStep 2: Generate original image folder\n')
        cls.generate_image_folder()

        print('\nStep 3: Generate GT image file\n')
        cls.generate_heatmap()

    @classmethod
    def generate_gt_numpy(cls):
        r"""
            랜드마크 좌표를 GT 폴더에 landmark_point_gt_numpy로 저장
            0~149: train 데이터
            150~299: test1 데이터
            300~399: test2 데이터
        """
        # 랜드마크 좌표를 저장할 numpy 배열
        junior_numpy = np.zeros((400, 19, 2))
        senior_numpy = np.zeros((400, 19, 2))
        average_numpy = np.zeros((400, 19, 2))

        # 원본 데이터 폴더
        original_data_gt_files_path = os.path.join(DataInfo.gp_path, r'original_data/AnnotationsByMD')
        junior_senior_folders = os.listdir(original_data_gt_files_path)
        junior_senior_folders.sort()

        # 원본 데이터 폴더에서 txt 파일을 읽어 numpy 배열로 저장
        # class는 저장하지 않고 건너뜀
        for junior_senior_folder in junior_senior_folders:
            original_files = os.listdir(os.path.join(original_data_gt_files_path, junior_senior_folder))
            original_files.sort()

            points_numpy = np.zeros((400, 19, 2))
            for original_file_index, original_file in enumerate(original_files):
                file = open(os.path.join(original_data_gt_files_path, junior_senior_folder, original_file))
                file_lines = file.readlines()
                file.close()

                for i, line in enumerate(file_lines):
                    if i < 19:
                        x, y = line.split(',')
                        x = int(x)
                        y = int(y)

                        points_numpy[original_file_index][i][0] = x
                        points_numpy[original_file_index][i][1] = y
                    else:
                        pass

            if junior_senior_folder[junior_senior_folder.index('_') + 1:] == "junior":
                junior_numpy = copy.deepcopy(points_numpy)
            else:
                senior_numpy = copy.deepcopy(points_numpy)

        # junior과 senior의 평균 구해서 numpy 배열로 저장
        for gt_file_index, [junior_points, senior_points] in enumerate(zip(junior_numpy, senior_numpy)):
            for landmark_index, [junior_point, senior_point] in enumerate(zip(junior_points, senior_points)):
                average_point_x = (junior_point[0] + senior_point[0]) / 2
                average_point_y = (junior_point[1] + senior_point[1]) / 2
                average_point = np.array([average_point_x, average_point_y])

                average_numpy[gt_file_index][landmark_index] = average_point

        # save
        gt_folder = os.path.join(DataInfo.gp_path, r'GT')
        os.makedirs(gt_folder, exist_ok=True)
        np.save(os.path.join(gt_folder, 'landmark_gt_numpy.npy'), average_numpy)

    @classmethod
    def generate_image_folder(cls):
        r"""
            원본 이미지를 train_val_data 폴더의 train, test 폴더에 복사
        """
        # 원본 이미지가 저장되어 있는 폴더
        original_images_folder = os.path.join(DataInfo.gp_path, r'original_data/original_images')
        images_folders = os.listdir(original_images_folder)
        images_folders.sort()

        # 이미지를 이동시킬 폴더
        train_val_data_folder = os.path.join(DataInfo.gp_path, r'train_test_data')
        if not os.path.exists(train_val_data_folder):
            os.makedirs(train_val_data_folder)

        for images_folder in images_folders:
            # train, val1, val2 폴더 각각 생성
            data_type = images_folder[:images_folder.index('_')]
            if not os.path.exists(os.path.join(train_val_data_folder, data_type, '0img')):
                os.makedirs(os.path.join(train_val_data_folder, data_type, '0img'))

            images = os.listdir(os.path.join(original_images_folder, images_folder))
            images.sort()

            # 이미지 복사
            for image in images:
                shutil.copy(os.path.join(original_images_folder, images_folder, image),
                            os.path.join(train_val_data_folder, data_type, '0img', image))

    @classmethod
    def generate_heatmap_with_numpy(cls, image_path, gt_numpy):
        heatmap_generator = AnisotropicLaplaceHeatmapGenerator()

        for i, gt in enumerate(tqdm(gt_numpy)):
            for j, landmark_point in enumerate(gt):
                heatmap_img = heatmap_generator.get_heatmap_image(landmark_point)
                landmark_gt_path = os.path.join(image_path, str(j + 1 + 100))
                os.makedirs(landmark_gt_path, exist_ok=True)

                image_name = str(i + 1).zfill(5)
                heatmap_img.save(os.path.join(landmark_gt_path, image_name + '.png'))

    @classmethod
    def generate_heatmap(cls):
        r"""
            generate gt image at 'train_test_data/
             - gt 이미지를 train, val1, val2 폴더에 랜드마크별로 저장

            params:
                new_size: 새로 생성할 GT 이미지 크기. [W, H]

                landmark_point = [W, H]
                landmark_gt_numpy:  0~149: train 데이터
                                    150~299: test1 데이터
                                    300~399: test2 데이터
        """

        train_image_path = os.path.join(DataInfo.gp_path, 'train_test_data/train')
        test1_image_path = os.path.join(DataInfo.gp_path, 'train_test_data/test1')
        test2_image_path = os.path.join(DataInfo.gp_path, 'train_test_data/test2')

        landmark_gt_numpy = np.load(os.path.join(DataInfo.gp_path, 'GT/landmark_gt_numpy.npy'))

        cls.generate_heatmap_with_numpy(train_image_path, landmark_gt_numpy[0:150])
        cls.generate_heatmap_with_numpy(test1_image_path, landmark_gt_numpy[150:300])
        cls.generate_heatmap_with_numpy(test2_image_path, landmark_gt_numpy[300:400])
