class Config:
    # ############################################################################
    # # 一些曾经的参数
    #
    # dataset_dir = 'Animal/dog/'
    #
    # coco_dir = '../coco'
    #
    # train_set = 'images/train2017'
    # val_set = 'images/val2017'
    #
    # coco_c = '../coco_c/images'
    #
    # # 场景分类器的超参数
    # batch_size = 1024
    # learning_rate = 0.04
    # epoch_num = 25
    #
    # # 场景分类器的训练集验证集的地址
    # classify_folder_train = '../coco_c/classify_folder_train_mini/*.jpg'
    # classify_folder_val = '../coco_c/classify_folder_val_mini/*.jpg'

    ################################################################################
    # 目前使用的参数

    # 数据生成器的参数
    coco_dir = '/mnt/disk1/yunzhe/coco'
    train_set = 'images/train2017'
    val_set = 'images/val2017'
    coco_c = '/mnt/disk1/yunzhe/coco_c/images'

    # 图片集的地址
    dataset_train = '/mnt/disk1/yunzhe/coco_c/images/all_train_images_4/*.jpg'
    dataset_val = '/mnt/disk1/yunzhe/coco_c/images/sample_val_images_2/*.jpg'

    # 代表集的地址
    sample_train = '../coco_c/classify_folder_train_mini/*.jpg'
    sample_val = '../coco_c/classify_folder_val_mini/*.jpg'

    # batch大小
    batch_size = 64

    # 代表集样本及标签
    sample_set = '/yolov5/data/sample.yaml'

    #########################################################
    # 全部数据集
    # dataset_train_list = ['/mnt/disk1/yunzhe/coco/images/train_person', '/mnt/disk1/yunzhe/kitti/yolo/images/',
    #                       '/mnt/disk1/yunzhe/NightOwls/images/train/',
    #                       '/mnt/disk1/yunzhe/caltech-pedestrian-dataset-to-yolo-format-converter/images/',
    #                       '/mnt/disk1/yunzhe/yolov4_crowdhuman/data/yolo/images/train/',
    #                       '/mnt/disk1/yunzhe/task_data/task0/images/', '/mnt/disk1/yunzhe/task_data/task2/images/',
    #                       '/mnt/disk1/yunzhe/task_data/task9/images/']

    # dataset_train_list = ['/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/caltech_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/coco_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/crowdhuman_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/kitti_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/night_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task0.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task1.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task2.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task3.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task4.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task5.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task7.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task8.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/task9.txt']

    # dataset_train_list = ['/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/caltech_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/coco_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/crowdhuman_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/kitti_train.txt',
    #                       '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/night_train.txt']

    dataset_train_list = ['/home/lion/yunzhe/TJU-DHD/dhd_traffic/train.txt']
    # dataset_train_list = ['/home/lion/yunzhe/calam/dhd_train_1.txt']

    # dataset_train_list = ['/home/lion/yunzhe/taskset/images/task1/', '/home/lion/yunzhe/taskset/images/task2/',
    #                       '/home/lion/yunzhe/taskset/images/task3/', '/home/lion/yunzhe/taskset/images/task4/',
    #                       '/home/lion/yunzhe/taskset/images/task5/', '/home/lion/yunzhe/taskset/images/task6/',
    #                       '/home/lion/yunzhe/taskset/images/task7/', '/home/lion/yunzhe/taskset/images/task8/']

    # labelset_train_list = ['/mnt/disk1/yunzhe/coco/labels/train_person', '/mnt/disk1/yunzhe/kitti/yolo/labels/',
    #                        '/mnt/disk1/yunzhe/NightOwls/labels/train/',
    #                        '/mnt/disk1/yunzhe/caltech-pedestrian-dataset-to-yolo-format-converter/labels/',
    #                        '/mnt/disk1/yunzhe/yolov4_crowdhuman/data/yolo/labels/train/',
    #                        '/mnt/disk1/yunzhe/task_data/task0/labels/', '/mnt/disk1/yunzhe/task_data/task2/labels/',
    #                        '/mnt/disk1/yunzhe/task_data/task9/labels/']

    # labelset_train_list = ['/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/labels/']

    labelset_train_list = ['/home/lion/yunzhe/taskset/labels/task1/', '/home/lion/yunzhe/taskset/labels/task2/', '/home/lion/yunzhe/taskset/labels/task3/', '/home/lion/yunzhe/taskset/labels/task4/', '/home/lion/yunzhe/taskset/labels/task5/', '/home/lion/yunzhe/taskset/labels/task6/', '/home/lion/yunzhe/taskset/labels/task7/', '/home/lion/yunzhe/taskset/labels/task8/']

    # dataset_val_list = ['/mnt/disk1/yunzhe/TJU-DHD/dhd_campus_train_images/dhd_campus/images/train/',
    #                     '/mnt/disk1/yunzhe/TJU-DHD/dhd_campus/images/val/',
    # dataset_val_list = ['/mnt/disk1/yunzhe/TJU-DHD/dhd_traffic/images/train/',
    #                     '/mnt/disk1/yunzhe/TJU-DHD/dhd_traffic/images/val/']
    # dataset_val_list = ['/home/lion/yunzhe/realworld_datasets/test/downtown/images/']
    # dataset_val_list = ['/home/lion/yunzhe/realworld_datasets/test/town/images/']

    # dataset_val_list = ['/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/caltech_val.txt',
    #                     '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/coco_val.txt',
    #                     '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/crowdhuman_val.txt',
    #                     '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/kitti_val.txt',
    #                     '/home/lion/yunzhe/cal_coco_kitti_city_crowd_night/night_val.txt']

    dataset_val_list = ['/home/lion/yunzhe/TJU-DHD/dhd_traffic/val.txt']

    # labelset_val_list = ['/mnt/disk1/yunzhe/TJU-DHD/dhd_campus/labels/train/',
    #                      '/mnt/disk1/yunzhe/TJU-DHD/dhd_campus/labels/val/',
    # labelset_val_list = ['/mnt/disk1/yunzhe/TJU-DHD/dhd_pedestrian/ped_traffic/labels/train/',
    #                      '/mnt/disk1/yunzhe/TJU-DHD/dhd_pedestrian/ped_traffic/labels/val/']
    # labelset_val_list = ['/home/lion/yunzhe/realworld_datasets/test/downtown/labels/']

    labelset_val_list = ['/home/lion/yunzhe/realworld_datasets/test/town/labels/']

    #########################################################
    # test参数

    # model_list = ['base_model/coco_c_7_2.yaml', 'base_model/coco_c_8_2.yaml',
    #               'base_model/coco_c_9_2.yaml', 'base_model/coco_c_10_2.yaml',
    #               'base_model/coco_c_11_2.yaml']

    # # Full acc
    # weight_list = ['./models_full/kitti.pt', './models_full/night.pt', './models_full/caltech.pt',
    #                './models_full/coco.pt', './models_full/crowdhuman.pt', './models_full/task0.pt',
    #                './models_full/task2.pt', './models_full/task9.pt']

    # weight_list = ['./models_full/kitti.pt', './models_full/caltech.pt', './models_full/night.pt',
    #                './models_full/task0.pt', './models_full/task2.pt', './models_full/coco.pt',
    #                './models_full/task9.pt', './models_full/crowdhuman.pt']

    # weight_list = ['./dhd-traffic/traffic-task1-e.pt',
    #                './dhd-traffic/traffic-task2-e.pt', './dhd-traffic/traffic-task3-e.pt',
    #                './dhd-traffic/traffic-task4-e.pt', './dhd-traffic/traffic-task5-e.pt',
    #                './dhd-traffic/traffic-task6-e.pt', './dhd-traffic/traffic-task7-e.pt',
    #                './dhd-traffic/traffic-task8-e.pt']

    weight_list = ['/home/lion/yunzhe/calam/dhd-traffic/traffic-all.pt', '/home/lion/yunzhe/calam/dhd-traffic/round1.pt', '/home/lion/yunzhe/calam/dhd-traffic/round2.pt', '/home/lion/yunzhe/calam/dhd-traffic/round3.pt', '/home/lion/yunzhe/calam/dhd-traffic/round4.pt', '/home/lion/yunzhe/calam/dhd-traffic/round5.pt', '/home/lion/yunzhe/calam/dhd-traffic/round6.pt']

    # weight_list = ['/home/lion/yunzhe/calam/dhd-traffic/traffic-all.pt',
    #                '/home/lion/yunzhe/calam/dhd-traffic/round1o.pt',
    #                '/home/lion/yunzhe/calam/dhd-traffic/round2o.pt',
    #                '/home/lion/yunzhe/calam/dhd-traffic/round3o.pt',
    #                '/home/lion/yunzhe/calam/dhd-traffic/round4o.pt']

    # weight_list = ['./models_v/kitti_v.pt', './models_v/caltech_v.pt', './models_v/night.pt',
    #                './models_v/coco.pt', './models_v/crowd_v.pt']

    # weight_list = ['./models_c8/caltech_c8_v.pt', './models_c8/coco.pt', './models_c8/crowdhuman_c8_v.pt',
    #                './models_c8/kitti_c8_v.pt', './models_c8/night_c8_v.pt',
    #                './models_c8/caltech_c8.pt', './models_c8/coco_c8.pt', './models_c8/crowd_c8.pt',
    #                './models_c8/kitti_c8.pt', './models_c8/night_c8.pt']

    # weight_list = ['./models_v/caltech_v.pt', './models_v/coco.pt', './models_v/crowd_v.pt',
    #                './models_v/kitti_v.pt', './models_v/night.pt', './all_v5s_v.pt']

    # c8
    # weight_list = ['./models_c8/kitti_c8.pt', './models_c8/night_c8.pt', './models_c8/caltech_c8.pt',
    #                './models_c8/coco_c8.pt', './models_c8/crowd_c8.pt', './models_c8/task0_c8.pt',
    #                './models_c8/task2_c8.pt', './models_c8/task9_c8.pt']

    # weight_list = ['./models_c8/kitti_c8.pt', './models_c8/caltech_c8.pt', './models_c8/night_c8.pt',
    #                './models_c8/task0_c8.pt', './models_c8/task2_c8.pt', './models_c8/coco_c8.pt',
    #                './models_c8/task9_c8.pt', './models_c8/crowd_c8.pt']

    baseline = 'base_model/all.pt'

    ##########################################################
    # 场景分类器参数
    learning_rate = 0.02
    epoch_num = 20

    ##########################################################
    # 算法超参数
    map_threshold = 0.42

    ##########################################################
    # 验证集大小
    val_set_len = 9050  # todo 应当作为参数传入，当前实现暂时以人工计算结果传入
    sample_len = 300  # 单个数据集sample的数量 500

    ##########################################################
    task_len = 20000

    ##########################################################
    # 批处理yaml文件
    yaml_dir = 'base_model/test.yaml'

    ##########################################################
    train_val_ratio = 0.05

    test_len = 2000
