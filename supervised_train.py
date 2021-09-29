from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os
setup_logger()

from detectron2.data.datasets import register_coco_instances, load_coco_json
register_coco_instances("soda10m.train", {}, "/mnt/disk/SODA10M/SSLAD-2D/labeled/annotations/instance_train.json", "/mnt/disk/SODA10M/SSLAD-2D/labeled/train")
register_coco_instances("soda10m.val", {}, "/mnt/disk/SODA10M/SSLAD-2D/labeled/annotations/instance_val.json", "/mnt/disk/SODA10M/SSLAD-2D/labeled/val")

#Evaluation with AP metric
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader


def do_train_and_eval():
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    evaluator = COCOEvaluator("soda10m.val", cfg, False, output_dir="./output/result")
    val_loader = build_detection_test_loader(cfg, "soda10m.val")
    inference_on_dataset(trainer.model, val_loader, evaluator)


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("/home/liyunzhe/detectron2-main/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
    cfg.DATASETS.TRAIN = ("soda10m.train",)
    # cfg.DATASETS.TEST = ("soda10m.val",)  # test on the val set
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.MODEL.DEVICE = 'cuda:3'
    cfg.MODEL.WEIGHT = "./output/model_final.pth"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    do_train_and_eval()
