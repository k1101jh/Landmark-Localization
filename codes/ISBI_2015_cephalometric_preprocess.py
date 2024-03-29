import os
import sys
import hydra
import logging
import shutil
import copy
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from hydra import compose, initialize
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from heatmap_generator.anisotropic_laplace_heatmap_generator import AnisotropicLaplaceHeatmapGenerator


OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def generate_ISBI_2015_Cephalometric_dataset():
    initialize(config_path="../configs", job_name="ISBI_preprocess")
    cfg = compose(config_name="config", overrides=["dataset=ISBI_2015_Cephalometric"]).dataset
    print(OmegaConf.to_yaml(cfg))

    def generate_landmark_point_file():
        r"""
        랜드마크 좌표를 GT 폴더에 landmark_point_gt_numpy로 저장
        0~149: train 데이터
        150~299: test1 데이터
        300~399: test2 데이터
        """
        # 랜드마크 좌표를 저장할 numpy 배열
        junior_ndarray = np.zeros((400, cfg.num_landmarks, 2))
        senior_ndarray = np.zeros((400, cfg.num_landmarks, 2))
        average_point_ndarray = np.zeros((400, cfg.num_landmarks, 2))
        resized_point_ndarray = np.zeros((400, cfg.num_landmarks, 2))

        resize_ratio_x = cfg.width / cfg.original_width
        resize_ratio_y = cfg.height / cfg.original_height

        # 원본 데이터 폴더
        junior_senior_folders = os.listdir(cfg.paths.original_data.annotations)
        junior_senior_folders.sort()

        # 원본 데이터 폴더에서 txt 파일을 읽어 numpy 배열로 저장
        # class는 저장하지 않고 건너뜀
        for junior_senior_folder in junior_senior_folders:
            label_txt_files = os.listdir(os.path.join(cfg.paths.original_data.annotations, junior_senior_folder))
            label_txt_files.sort()

            points_ndarray = np.zeros((400, cfg.num_landmarks, 2))

            # Read annotation txt files
            for original_file_index, original_file in enumerate(label_txt_files):
                file = open(os.path.join(cfg.paths.original_data.annotations, junior_senior_folder, original_file))
                file_lines = file.readlines()
                file.close()

                for i, line in enumerate(file_lines):
                    if i < cfg.num_landmarks:
                        x, y = line.split(",")
                        x = int(x)
                        y = int(y)

                        points_ndarray[original_file_index][i][0] = x
                        points_ndarray[original_file_index][i][1] = y
                    else:
                        break

            if junior_senior_folder[junior_senior_folder.index("_") + 1 :] == "junior":
                junior_ndarray = copy.deepcopy(points_ndarray)
            else:
                senior_ndarray = copy.deepcopy(points_ndarray)

        # junior과 senior의 평균 구해서 numpy 배열로 저장
        for gt_file_index, [junior_points, senior_points] in enumerate(zip(junior_ndarray, senior_ndarray)):
            for landmark_index, [junior_point, senior_point] in enumerate(zip(junior_points, senior_points)):
                average_point_x = (junior_point[0] + senior_point[0]) / 2
                average_point_y = (junior_point[1] + senior_point[1]) / 2
                average_point = np.array([average_point_x, average_point_y])

                average_point_ndarray[gt_file_index][landmark_index] = average_point

                resized_point = np.array([average_point_x * resize_ratio_x, average_point_y * resize_ratio_y])
                resized_point_ndarray[gt_file_index][landmark_index] = resized_point

        # save
        landmark_point_dict = {}
        landmark_point_dict["train"] = average_point_ndarray[:150].tolist()
        landmark_point_dict["test1"] = average_point_ndarray[150:300].tolist()
        landmark_point_dict["test2"] = average_point_ndarray[300:].tolist()
        resized_landmark_point_dict = {}
        resized_landmark_point_dict["train"] = resized_point_ndarray[:150].tolist()
        resized_landmark_point_dict["test1"] = resized_point_ndarray[150:300].tolist()
        resized_landmark_point_dict["test2"] = resized_point_ndarray[300:].tolist()
        landmark_point_df = pd.DataFrame.from_dict(landmark_point_dict, orient="index")
        landmark_point_df = landmark_point_df.transpose()
        landmark_point_df.to_pickle(cfg.paths.landmark_point_pkl)
        resized_landmark_point_df = pd.DataFrame.from_dict(resized_landmark_point_dict, orient="index")
        resized_landmark_point_df = resized_landmark_point_df.transpose()
        resized_landmark_point_df.to_pickle(cfg.paths.resized_landmark_point_pkl)

    def copy_input_images():
        r"""
        원본 이미지를 dataset 폴더에 복사
        """
        source_image_dirs = [
            cfg.paths.original_data.train_images,
            cfg.paths.original_data.test1_images,
            cfg.paths.original_data.test2_images,
        ]
        dest_image_dirs = [cfg.paths.train.images, cfg.paths.test1.images, cfg.paths.test2.images]

        for source_dir, dest_dir in tqdm(zip(source_image_dirs, dest_image_dirs)):
            os.makedirs(dest_dir, exist_ok=True)

            images = os.listdir(source_dir)
            images.sort()

            # 이미지 복사
            for image in images:
                shutil.copy(
                    os.path.join(source_dir, image),
                    os.path.join(dest_dir, image),
                )

    def generate_heatmap():
        r"""
        generate gt image at 'train_test_image/
         - gt 이미지를 train, test1, test2 폴더에 랜드마크별로 저장

        params:
            new_size: 새로 생성할 GT 이미지 크기. [W, H]

            landmark_point = [W, H]
            landmark_gt_numpy:  0~149: train 데이터
                                150~299: test1 데이터
                                300~399: test2 데이터
        """
        heatmap_generator = instantiate(cfg.heatmap_generator)

        landmark_point_dict = pd.read_pickle(cfg.paths.resized_landmark_point_pkl).to_dict()

        for dataset_type in cfg.dataset_types:
            heatmap_path = cfg.paths[dataset_type].heatmaps
            all_image_landmark_points = landmark_point_dict[dataset_type]
            for i, landmark_points in tqdm(all_image_landmark_points.items()):
                if landmark_points:
                    for j, landmark_point in enumerate(landmark_points):
                        heatmap_image = heatmap_generator.get_heatmap_image(landmark_point)
                        heatmap_save_path = os.path.join(heatmap_path, "{:0>2d}".format(j + 1))
                        os.makedirs(heatmap_save_path, exist_ok=True)

                        heatmap_image.save(os.path.join(heatmap_save_path, f"{str(i + 1).zfill(3)}.png"))

    os.makedirs(cfg.paths.dataset, exist_ok=True)
    logging.info("Start generating gt numpy files...")
    generate_landmark_point_file()

    logging.info("Start copying images to dataset folder...")
    copy_input_images()

    logging.info("Start generating heatmap image files...")
    generate_heatmap()


if __name__ == "__main__":
    generate_ISBI_2015_Cephalometric_dataset()
