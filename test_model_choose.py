import os
import joblib
import random
import torch

from sample_node import sample_node
from config import Config


def sample_list_gen(task_folder_list, sample_list_name):
    # sample generation

    sample_index = 0
    sample_list = []

    for task_folder_index in range(len(task_folder_list)):
        f = open(task_folder_list[task_folder_index])
        file_list = f.readlines()
        random.shuffle(file_list)
        f.close()

        file_index = 0

        length = len(file_list)

        for file_obj_index in range(len(file_list)):
            if file_index < length:
                try:
                    file_obj = file_list[file_obj_index].strip("\n")
                    file_path = file_obj
                    file_dir, file_name = os.path.split(file_path)

                    label_path = os.path.join(os.path.dirname(file_dir), 'labels', file_name[:-4] + '.txt')

                    if os.path.exists(label_path) and os.path.getsize(label_path):  # check that tha label exists and is not empty
                        # print(os.path.getsize(label_path))
                        sample = sample_node(sample_index, file_path, label_path)
                        sample.sample_evalution()  # performance evauation
                        # print('check result: ', sample.sample_check())
                        sample_list.append(sample)
                        sample_index = sample_index + 1
                        file_index = file_index + 1
                        print(file_index)

                except Exception as e:
                    print(e)
                    continue

    # print(sample_list[300].feature)
    joblib.dump(sample_list, sample_list_name)

if __name__ == "__main__":
    sample_list_gen(Config.dataset_val_list, 'test_sample.s')
