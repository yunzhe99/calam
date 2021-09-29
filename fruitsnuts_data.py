from detectron2.data.datasets import register_coco_instances
register_coco_instances("soda10m", {}, "./data/trainval.json", "./data/images")
