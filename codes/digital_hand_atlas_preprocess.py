import os
import sys
import hydra
import logging
import shutil
import imagesize
import copy
import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from hydra.utils import instantiate
from hydra import compose, initialize
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from multiprocessing import current_process
from tqdm.contrib.concurrent import process_map

from interface.functions import *

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)

HEATMAP_SCALE = 4


def generate_heatmap_multiprocessing(heatmap_generator, landmark_points, heatmap_save_dir_path):
    process_idx = current_process()._identity[0] - 1
    # heatmap 생성
    for img_filename, landmark_points in tqdm(
        landmark_points["test"].items(),
        desc=f"process {process_idx} generating heatmap...",
        position=process_idx + 1,
        leave=True,
    ):
        for i, landmark_point in enumerate(landmark_points):
            heatmap_save_path = os.path.join(heatmap_save_dir_path, f"{(i + 1):0>2d}", f"{img_filename}.png")
            heatmap_image = heatmap_generator.get_heatmap_image(landmark_point)

            heatmap_image.save(heatmap_save_path)


def generate_digital_hand_atlas_dataset():
    initialize(config_path="../configs", job_name="Digital_Hand_Atlas_preprocess")
    cfg = compose(config_name="config", overrides=["dataset=Digital_Hand_Atlas"]).dataset
    print(OmegaConf.to_yaml(cfg))

    def copy_input_images():
        """setup 폴더 생성하고 원본 이미지 복사

        Args:
            original_data_dir (_type_): _description_
            dest_dir (_type_): _description_
        """

        logging.info("Start copying images to dataset dir..")

        if (not cfg.overwrite) and os.path.exists(cfg.paths.dataset):
            logging.info("Directory already exists!")
            return
        else:
            original_image_path = os.path.join(cfg.paths.original_data.images)
            img_class_dir_paths = glob.glob(original_image_path + "/*")
            # {setup}/images 폴더 생성
            for setup in tqdm(cfg.setup_list, desc="Splitting data...", position=0, leave=True):
                setup_dir_path = os.path.join(cfg.paths.original_data.setup_dir, str(setup))

                dest_dir_path = os.path.join(cfg.paths.dataset, str(setup), "images")
                os.makedirs(dest_dir_path, exist_ok=True)

                # test.txt 파일 읽어서 이미지 파일명 리스트 가져오기
                txt_file_path = os.path.join(setup_dir_path, "test.txt")
                with open(txt_file_path, "r") as file:
                    img_files = file.readlines()

                for i in range(len(img_files)):
                    img_files[i] = img_files[i][:-1]

                # 원본 파일 중에서 이미지 찾아서 setup 데이터 폴더에 복사
                for img_class_dir_path in tqdm(img_class_dir_paths, position=1, leave=False):
                    age_dir_paths = glob.glob(img_class_dir_path + "/*")
                    for age_dir_path in age_dir_paths:
                        age_dir_img_files = os.listdir(age_dir_path)
                        for img_file in age_dir_img_files:
                            if os.path.splitext(img_file)[0] in img_files:
                                shutil.copy(
                                    os.path.join(age_dir_path, img_file),
                                    os.path.join(dest_dir_path, img_file),
                                )

            logging.info("Copying images to dataset dir completed!")

    def save_original_image_size():
        """
        원본 이미지 파일 크기를 저장 (width, height)
        """
        logging.info("Start saving original image size...")

        if (not cfg.overwrite) and os.path.exists(cfg.paths.original_size_pkl):
            logging.info("File already exists!")
            return
        else:
            image_sizes = {}
            for setup in cfg.setup_list:
                image_dir_path = os.path.join(cfg.paths.dataset, str(setup), "images")

                # 파일을 읽어서 이미지 크기 알아내기
                image_list = os.listdir(image_dir_path)
                for image_filename in tqdm(image_list, desc="Getting image size..."):
                    width, height = imagesize.get(os.path.join(image_dir_path, image_filename))
                    image_sizes[os.path.splitext(image_filename)[0]] = [width, height]

            df = pd.DataFrame(image_sizes)
            df.to_pickle(cfg.paths.original_size_pkl)

            logging.info("Saving original image size completed!")

    def generate_landmark_point_file():
        def read_annotation_file(annotation_file_path):
            """
            annotation file을 읽어서 다음 형식의 dict 반환
            key: image name
            value: landmark point list
            """
            with open(annotation_file_path, "r") as file:
                lines = file.readlines()

            landmark_points_dict = {}

            for line in lines:
                image_name, landmark_points = line.strip().split(":")
                image_name = image_name.strip('"').split(".")[0]
                landmark_points = landmark_points.strip(")").split("(")[1:]
                landmark_point_list = []
                for landmark_point in landmark_points:
                    landmark_point = landmark_point.split(",")
                    landmark_point_list.append([int(pos_str) for pos_str in landmark_point[:2]])
                landmark_points_dict[image_name] = landmark_point_list

            return landmark_points_dict

        logging.info("Start generating landmark point data...")

        # 파일이 이미 존재하는지 확인
        if not cfg.overwrite and bool(
            os.path.exists(cfg.paths.landmark_point_pkl) and os.path.exists(cfg.paths.resized_landmark_point_pkl)
        ):
            logging.info("File already exists!")
            return
        else:
            # annotation file을 읽어서 랜드마크 위치 알아내기
            annotation_landmark_points_dict = read_annotation_file(cfg.paths.original_data.annotations)

            # 원본 이미지 크기 dataframe을 dict로 불러오기
            original_image_size_dict = pd.read_pickle(cfg.paths.original_size_pkl).to_dict()

            # setup에 따라 데이터를 3개 세트로 분류
            landmark_point_dict = {}
            resized_landmark_point_dict = {}

            for setup in tqdm(cfg.setup_list, desc="setup number:", position=0):
                setup_dir_path = setup_dir_path = os.path.join(cfg.paths.original_data.setup_dir, str(setup))
                landmark_point_dict[setup] = {}
                resized_landmark_point_dict[setup] = {}

                for dataset_type in cfg.dataset_types:
                    # txt 파일 읽어서 해당 dataset type에 맞는 이미지 파일명 리스트 가져오기
                    txt_file_path = os.path.join(setup_dir_path, dataset_type + ".txt")

                    with open(txt_file_path, "r") as file:
                        img_filenames = file.readlines()
                        img_filenames.sort()

                    for i in range(len(img_filenames)):
                        img_filenames[i] = img_filenames[i][:-1]

                    # 랜드마크 포인트를 dict에 ndarray로 저장
                    landmark_point_dict[setup][dataset_type] = {}
                    resized_landmark_point_dict[setup][dataset_type] = {}

                    for img_filename in img_filenames:
                        landmark_points = annotation_landmark_points_dict[img_filename]
                        landmark_point_dict[setup][dataset_type][img_filename] = np.zeros((cfg.num_landmarks, 2))
                        resized_landmark_point_dict[setup][dataset_type][img_filename] = np.zeros(
                            (cfg.num_landmarks, 2)
                        )
                        for landmark_idx, landmark_point in enumerate(landmark_points):
                            # width, height
                            resized_landmark_point = np.array(
                                [
                                    landmark_point[0] * (cfg.width / original_image_size_dict[img_filename][0]),
                                    landmark_point[1] * (cfg.height / original_image_size_dict[img_filename][1]),
                                ]
                            )
                            landmark_point_dict[setup][dataset_type][img_filename][landmark_idx] = np.flip(
                                np.array(landmark_point)
                            )
                            resized_landmark_point_dict[setup][dataset_type][img_filename][
                                landmark_idx
                            ] = resized_landmark_point

            # save landmark point files
            landmark_point_df = pd.DataFrame(landmark_point_dict)
            landmark_point_df.to_pickle(cfg.paths.landmark_point_pkl)

            resized_landmark_point_df = pd.DataFrame(resized_landmark_point_dict)
            resized_landmark_point_df.to_pickle(cfg.paths.resized_landmark_point_pkl)

            logging.info("Landmark point data generation completed")

    def generate_heatmaps():
        """
        이미지 한 장은 세 개의 setup 파일에 다 들어감
        test setup 파일을 읽어서 해당 setup/test/heatmap_image 폴더에 넣고
        다른 setup의 train/heatmap_image 폴더에 넣기
        """

        logging.info("Start generating heatmap images...")

        # landmark point가 저장된 파일 불러와서
        # setup 끼리 묶기
        landmark_point_dict = pd.read_pickle(cfg.paths.resized_landmark_point_pkl).to_dict()
        landmark_points_per_setup = [landmark_point_dict[setup] for setup in cfg.setup_list]

        # heatmap 저장 폴더 생성
        dir_exists = False
        heatmap_save_dir_paths = [cfg.paths.setup1.heatmaps, cfg.paths.setup2.heatmaps, cfg.paths.setup3.heatmaps]
        for heatmap_save_dir_path in heatmap_save_dir_paths:
            if os.path.exists(heatmap_save_dir_path):
                dir_exists = True

            for i in range(cfg.num_landmarks):
                os.makedirs(
                    os.path.join(heatmap_save_dir_path, f"{(i + 1):0>2d}"),
                    exist_ok=True,
                )

        if bool(not cfg.overwrite) and dir_exists:
            logging.info("Heatmap dir already exists!")
        else:
            heatmap_generator = instantiate(cfg.heatmap_generator)
            results = process_map(
                generate_heatmap_multiprocessing,
                [heatmap_generator] * len(cfg.setup_list),
                landmark_points_per_setup,
                heatmap_save_dir_paths,
                desc="overall progress",
            )
            logging.info("Heatmap image generation completed")

    # train, test 파일 옮기기
    copy_input_images()

    # 원본 이미지 파일 크기를 저장
    save_original_image_size()

    # landmark point를 파일로 저장
    # resize된 landmark point도 저장
    generate_landmark_point_file()

    # heatmap 생성(resized)
    generate_heatmaps()


if __name__ == "__main__":
    generate_digital_hand_atlas_dataset()
