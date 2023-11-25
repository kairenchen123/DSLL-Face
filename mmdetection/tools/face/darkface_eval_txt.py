import cv2
import mmcv
import os
import csv
import numpy
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import matplotlib.pyplot as plt

# 指定模型的配置文件和 checkpoint 文件路径
config_file = '/mnt/data2/ckr/mmdetection/local_config/yolox/yolox_darkface/yolox_l_4layer_internimage_darkface.py'
checkpoint_file = '/mnt/data2/ckr/mmdetection/work_dirs/darkface/intern/yolox_use4layer_inception13_DIoULoss/best_pascal_voc_mAP_epoch_300.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# 根据配置文件和 checkpoint 文件构建模型
img_root = '/mnt/data2/ckr/mmdetection/data/test_mem'
predict_root = '/mnt/data2/ckr/mmdetection/work_dirs/pred/res'
if not os.path.exists(predict_root):
    os.makedirs(predict_root)
txt_writer = ""
img_list = os.listdir(img_root)

for imageFile in img_list:
    txt_to_write = ""
    image = os.path.join(img_root, imageFile)
    pred = inference_detector(model, image)
    nums = len(pred.pred_instances.bboxes)
    for i in range(nums):
        x1, y1, x2, y2, score = pred.pred_instances.bboxes[i][0], pred.pred_instances.bboxes[i][1], \
                                pred.pred_instances.bboxes[i][2], pred.pred_instances.bboxes[i][3], \
                                pred.pred_instances.scores[i]
        # 构建包含坐标和分数的字符串
        score = round(score.item(), 4)
        # x1 = "{:.4f}".format(x1)
        x1 = round(x1.item(), 4)
        y1 = round(y1.item(), 4)
        x2 = round(x2.item(), 4)
        y2 = round(y2.item(), 4)

        bbox = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(score)
        txt_to_write += bbox + "\n"
    name = imageFile[:imageFile.rfind('.')]
    file_name = name + '.txt'
    file_path = os.path.join(predict_root, file_name)
    with open(file_path, 'w', encoding='utf-8') as fw:
        fw.write(txt_to_write)