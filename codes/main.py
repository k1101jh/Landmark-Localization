import os
import sys
import torch

from enums.dataset_enum import DatasetEnum
from interface.functions import *
from dataset_generator.dataset_generator import DatasetGenerator
from model.model import Model
from data_info.data_info import DataInfo


def run():
    # generate dataset
    if get_yes_no_input("Generate dataset?", False):
        DatasetGenerator.generate_dataset()

    while True:
        # select gpu num
        if torch.cuda.device_count() > 0:
            device_num = get_int_input("select device num", 0, torch.cuda.device_count())
        else:
            print("No GPUs!")
            sys.exit(0)

        # Input model folder name
        print("model list: ")
        if os.path.exists(DataInfo.saved_model_folder_path):
            for folder in os.listdir(DataInfo.saved_model_folder_path):
                print('  ', folder)
        else:
            print('None')
        model_name = input("\nInput model folder name >> ")
        model_folder_path = os.path.join(DataInfo.saved_model_folder_path, model_name)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        else:
            print("model exists!")

        # Check use tensorboard
        use_tensorboard = get_yes_no_input("Use tensorboard?", True)

        # print configure
        print("\nConfigures: ")
        print("GPU num: ", device_num)
        print("Model name: ", model_name)
        print("Model path: ", model_folder_path)
        print("Use tensorboard: ", use_tensorboard)

        if not get_yes_no_input("change configure?", False):
            break

    model = Model(device_num, model_name, model_folder_path, use_tensorboard)

    # Train model
    if get_yes_no_input("Train model?", False):
        epochs = 1000
        lr = 1e-4
        perturbation_percentage = []

        # Check load model
        load = False
        if len(os.listdir(model_folder_path)) > 0:
            if get_yes_no_input("Load last model?", True):
                load = True

        if not load:
            while True:
                # set perturbator percentage
                perturbation_percentage = get_perturbation_percentage()

                # Input model configurations
                print("\nModel configurations")
                print("epochs: ", epochs, " learning rate: ", lr)
                if get_yes_no_input("Change model configurations?", False):
                    while True:
                        try:
                            epochs = int(input("Input Epochs>> "))
                            lr = float(input("Input learning rate>>"))
                            break
                        except Exception as e:
                            print(e)

                print("Epochs: ", epochs)
                print("Learning rate: ", lr)
                print("Perturbator percentage: ", perturbation_percentage)

                if not get_yes_no_input("change configure?", False):
                    break

        model.train(load=load, perturbation_percentage=perturbation_percentage, num_epochs=epochs, lr=lr)

    elif get_yes_no_input("Test model?", False):
        # Test model
        test_model_list = os.listdir(model_folder_path)
        test_model_name = test_model_list[-1]
        while True:
            if get_yes_no_input("Select test model manually? (default: last model)", False):
                test_model_name = str(input("Input test model name>> "))
                if test_model_name in test_model_list:
                    break
                else:
                    print("No such file")
            else:
                break

        print("Select test data. ", end='')
        for member in DatasetEnum:
            print(member.name, ': ', member.value, end=' ')
        test_dataset_type = DatasetEnum(get_int_input(" >>", 0, len(DatasetEnum.__members__)))

        save_result_images = get_yes_no_input("Save result images?", True)

        test_model_path = os.path.join(model_folder_path, test_model_name)

        print("Test_model_name: ", test_model_name)
        print("Test model path: ", test_model_path)
        print("Test dataset type: ", test_dataset_type.name)
        print("Save Test result images: ", save_result_images)
        model.test(test_model_path, save_result_images=save_result_images, dataset_type=test_dataset_type)


if __name__ == '__main__':
    os.system("stty -echo")
    run()
