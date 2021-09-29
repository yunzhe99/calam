import os
import torch
import numpy as np

test_dir = '/mnt/disk/TJU-DHD/dhd_traffic/trainset/images/'

def test_load():
    full_path_list = []
    for test_path in os.listdir(test_dir):
        full_path = os.path.join(test_dir, test_path)
        full_path_list.append(full_path)
    return full_path_list


if __name__ == "__main__":

    test_list = test_load()
    test_batch = test_list[:10]
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./dhd-traffic/traffic_all_big.pt',)
    results = model(test_batch)
    results = results.pandas().xyxyn
    conf_list = []
    for result in results:
        conf = np.mean(np.array(result['confidence']))
        conf_list.append(conf)
    conf_total = np.mean(conf_list)
    print(conf_total)
