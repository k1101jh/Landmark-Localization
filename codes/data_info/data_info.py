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

    # model save path
    model_save_path = os.path.join(gp_path, "saved_models")

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
