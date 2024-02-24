import os
import time
import hydra
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2 as cv
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader


OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def resize_landmark_points(landmark_points, image_size, original_image_size):
    # [width, height]
    resize_ratio = [original_image_size[0] / image_size[0], original_image_size[1] / image_size[1]]

    resized_landmark_points = []
    for landmark_point in landmark_points:
        resized_landmark_points.append([landmark_point[0] * resize_ratio[0], landmark_point[1] * resize_ratio[1]])

    return resized_landmark_points


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def test(cfg: DictConfig):
    config_yaml = OmegaConf.to_yaml(cfg)
    print(config_yaml)

    ## device 설정
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("\tGPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    ## generate result image save path
    assert cfg.test.load_model_path != None

    result_dir = os.path.join(os.path.dirname(cfg.test.load_model_path), "test_result")

    if cfg.test.save_result_images:
        landmark_point_image_dir = os.path.join(result_dir, "landmark_point_images")
        landmark_point_result_image_dir = os.path.join(landmark_point_image_dir, "results")
        landmark_point_gt_image_dir = os.path.join(landmark_point_image_dir, "gt")
        landmark_point_comp_image_dir = os.path.join(landmark_point_image_dir, "compare")
        result_heatmap_dir = os.path.join(result_dir, "heatmaps")
        os.makedirs(landmark_point_image_dir, exist_ok=True)
        os.makedirs(landmark_point_result_image_dir, exist_ok=True)
        os.makedirs(landmark_point_gt_image_dir, exist_ok=True)
        os.makedirs(landmark_point_comp_image_dir, exist_ok=True)
        os.makedirs(result_heatmap_dir, exist_ok=True)

        for image_idx in range(0, 400):
            os.makedirs(os.path.join(result_heatmap_dir, str(image_idx + 1)), exist_ok=True)

    ## dataset
    dataset = instantiate(cfg.dataset, cfg.test_dataset_type, cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=cfg.test.num_workers)
    landmark_point_dict = pd.read_pickle(cfg.dataset.paths.landmark_point_pkl).to_dict()

    # load_model
    model = instantiate(cfg.model.constructor)
    loaded_model = torch.load(cfg.test.load_model_path, map_location=device)
    model.load_state_dict(loaded_model["model"])
    model = model.to(device)
    model.eval()

    # load gt numpy
    # [image_num, width, height]
    landmark_point_dict = pd.read_pickle(cfg.dataset.paths.landmark_point_pkl).to_dict()
    landmark_point_dict = landmark_point_dict[cfg.test_dataset_type]

    distance_list = []
    image_idx = 0

    for inputs, labels, meta_datas in tqdm(dataloader, desc="iterations", position=0, leave=True):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # collate_fn을 설정하거나, original_size를 transpose하거나
        meta_datas["original_size"] = np.array([t.numpy() for t in meta_datas["original_size"]]).transpose()

        outputs = model(inputs)
        outputs = outputs.data

        for idx in range(inputs.size(0)):
            # 히트맵으로 랜드마크 위치 추정하기
            heatmaps = outputs[idx].detach().cpu()
            input_image_path = meta_datas["image_path"][idx]
            input_image_name = os.path.basename(input_image_path)
            original_image_size = meta_datas["original_size"][idx]
            gt_landmark_points = landmark_point_dict[image_idx]
            image_idx += 1
            landmark_points = []

            for landmark_idx in range(cfg.dataset.num_landmarks):
                landmark_heatmap = heatmaps[landmark_idx]

                # np.where의 결과로 [[axis=0 인덱스 번호], [axis=1 인덱스 번호]]를 얻음
                # 여기에 mean(axis=1)을 적용하면 [y, x]를 얻음
                # [x, y]로 바꿔서 저장하기
                landmark_heatmap_max = np.array(
                    np.where(landmark_heatmap > landmark_heatmap.max() * cfg.test.heatmap_threshold)
                )
                landmark_point = landmark_heatmap_max.mean(axis=1)
                landmark_points.append([landmark_point[1], landmark_point[0]])

            # 랜드마크 크기를 원래 이미지 크기에 맞게 리사이즈
            landmark_points = resize_landmark_points(
                landmark_points, [cfg.dataset.width, cfg.dataset.height], original_image_size
            )

            # gt와 추정값의 거리 계산
            dists = dataset.calc_dists(landmark_points, gt_landmark_points)
            distance_list.append(dists)

            # 결과 이미지 저장
            if cfg.test.save_result_images:
                # 랜드마크 포인트 이미지 저장
                result_landmark_image = cv.imread(input_image_path)
                gt_landmark_image = deepcopy(result_landmark_image)
                comp_landmark_image = deepcopy(result_landmark_image)

                for landmark_idx in range(cfg.dataset.num_landmarks):
                    landmark_point = landmark_points[landmark_idx]
                    gt_landmark_point = gt_landmark_points[landmark_idx]
                    cv.circle(
                        result_landmark_image, (int(landmark_point[0]), int(landmark_point[1])), 17, (255, 0, 0), -1
                    )
                    cv.circle(
                        gt_landmark_image, (int(gt_landmark_point[0]), int(gt_landmark_point[1])), 17, (0, 0, 255), -1
                    )
                    cv.circle(
                        comp_landmark_image, (int(landmark_point[0]), int(landmark_point[1])), 17, (255, 0, 0), -1
                    )
                    cv.circle(
                        comp_landmark_image, (int(gt_landmark_point[0]), int(gt_landmark_point[1])), 17, (0, 0, 255), -1
                    )

                cv.imwrite(os.path.join(landmark_point_result_image_dir, input_image_name), result_landmark_image)
                cv.imwrite(os.path.join(landmark_point_gt_image_dir, input_image_name), gt_landmark_image)
                cv.imwrite(os.path.join(landmark_point_comp_image_dir, input_image_name), comp_landmark_image)

                # 히트맵 이미지 저장
                for landmark_idx in range(cfg.dataset.num_landmarks):
                    landmark_heatmap = heatmaps[landmark_idx]
                    plt.imsave(
                        os.path.join(
                            result_heatmap_dir,
                            str(landmark_idx + 1),
                            input_image_name,
                        ),
                        landmark_heatmap,
                        cmap="gray",
                    )

    # calculate accuracy of all landmarks
    calc_statistics(distance_list, save_path=os.path.join(result_dir, cfg.test.statistic_save_filename))

    # save distance
    distance_df = pd.DataFrame(distance_list)
    distance_df.to_pickle(os.path.join(result_dir, cfg.test.distance_save_filename))


def calc_statistics(landmark_dist_list, save_path=None):
    # calculate accuracy by landmarks
    distance_ndarray = np.array(landmark_dist_list)
    num_landmarks = distance_ndarray.shape[1]

    dist_limits = [2, 2.5, 3, 4]
    accuracy_ndarray = np.zeros((num_landmarks, len(dist_limits)))

    for i in range(num_landmarks):
        for dist_idx in range(len(dist_limits)):
            accuracy_ndarray[i][dist_idx] = np.mean(distance_ndarray[:, i] < dist_limits[dist_idx])

    if save_path:
        output_save_file = open(save_path, "w")
    else:
        output_save_file = None

    def print_f(str, **kwargs):
        print(str, file=output_save_file, **kwargs)

    ## 랜드마크 별 거리 차이 평균, 중앙값, 표준편차 출력

    print_f("\t", end="")
    for dist_limit in dist_limits:
        print_f(f"\t{dist_limit}", end="")
    print_f("\tmean\tmedian\tstd", end="\n")
    for i in range(num_landmarks):
        print_f(f"point num: {(i + 1):2d}", end="\t")
        for j in range(accuracy_ndarray.shape[1]):
            print_f(f"{accuracy_ndarray[i][j] * 100:.4f}", end="\t")

        print_f(f"{np.mean(distance_ndarray[:, i]):.4f}", end="\t")
        print_f(f"{np.median(distance_ndarray[:, i]):.4f}", end="\t")
        print_f(f"{np.std(distance_ndarray[:, i]):.4f}", end="\t")
        print_f("", end="\n")

    ## 전체 평균, 중앙값, 표준편차, 정확도
    print_f("", end="\n")
    print_f(f"mean: {np.mean(distance_ndarray):.4f}", end="\n")
    print_f(f"median: {np.median(distance_ndarray):.4f}", end="\n")
    print_f(f"std: {np.std(distance_ndarray):.4f}", end="\n")
    print_f("accuracy:", end="\n")
    for dist_idx in range(len(dist_limits)):
        print_f(f"{dist_limits[dist_idx]}mm: ", end="")
        print_f(f"\t{np.mean(accuracy_ndarray[:, dist_idx]) * 100:.4f}", end="\n")


if __name__ == "__main__":
    test()
