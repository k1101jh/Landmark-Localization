"""
2020-03-17 작성
Kang Jun Hyeok

데이터에 대한 정보들이 담긴 파일
"""

import os


class DataInfo:
    # codes folder
    p_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    # JBNU_medical_data folder
    gp_path = os.path.dirname(p_path)

    # paths
    original_data_folder_path = os.path.join(gp_path, "original_data")
    original_gt_folder_path = os.path.join(original_data_folder_path, 'AnnotationsByMD')
    raw_image_folder_path = os.path.join(original_data_folder_path, "RawImage")
    train_test_image_folder_path = os.path.join(gp_path, "train_test_image")
    saved_model_folder_path = os.path.join(gp_path, "saved_models")
    landmark_gt_numpy_folder_path = os.path.join(gp_path, 'GT')
    landmark_gt_numpy_path = os.path.join(landmark_gt_numpy_folder_path, 'landmark_gt_numpy.npy')
    tensorboard_path = os.path.join(gp_path, "runs")

    # test result paths
    result_image_folder_path = os.path.join(gp_path, 'images', 'result_image')
    output_image_save_path = os.path.join(result_image_folder_path, 'output')
    gt_image_save_path = os.path.join(result_image_folder_path, 'gt')

    result_heatmap_image_folder_path = os.path.join(gp_path, 'images', 'result_heatmap_image')
    output_heatmap_image_save_path = os.path.join(result_heatmap_image_folder_path, 'output')
    gt_heatmap_image_save_path = os.path.join(result_heatmap_image_folder_path, 'gt')

    # width, height
    original_image_size = [1935, 2400]
    resized_image_size = [640, 800]

    # perturbator parameters
    boxing_scheme = 1

    # model parameters
    landmark_class_num = 19
    batch_size = 2
    pow_heatmap = 7
    perturbation_scheme = [1, 1]

    is_attention_unet = True

    # test parameters
    heatmap_remain_ratio = 0.85

    save_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
