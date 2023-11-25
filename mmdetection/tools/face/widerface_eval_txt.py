import cv2
import mmcv
import os
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
from PIL import Image
import matplotlib.pyplot as plt

# 指定模型的配置文件和 checkpoint 文件路径
config_file = '/home/gzhu2023/gzhu2023/ckr/mmdetection/local_config/yolo/yolov3_tina_widerface.py'
checkpoint_file = '/home/gzhu2023/gzhu2023/ckr/mmdetection/work_dirs/kernel_11/best_pascal_voc_mAP_epoch_49.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# 根据配置文件和 checkpoint 文件构建模型
eval_root = '/home/gzhu2023/gzhu2023/ckr/mmdetection/data/widerface/WIDER_val/images'
eval_list = os.listdir(eval_root)
predict_root = '/home/gzhu2023/gzhu2023/ckr/mmdetection/data/pred'


for dire in eval_list:
    text_root = os.path.join(predict_root, dire)
    os.makedirs(text_root)
    img_root = os.path.join(eval_root, dire)
    img_list = os.listdir(img_root)
    for imageFile in img_list:
        txt_writer = ""
        txt_to_write = ""
        image = os.path.join(img_root, imageFile)
        pred = inference_detector(model, image)
        nums = len(pred.pred_instances.bboxes)
        for i in range(nums):
            x1, y1, x2, y2, score = int(pred.pred_instances.bboxes[i][0]), int(pred.pred_instances.bboxes[i][1]), \
                                    int(pred.pred_instances.bboxes[i][2]), int(pred.pred_instances.bboxes[i][3]), \
                                    pred.pred_instances.scores[i]
            # 构建包含坐标和分数的字符串
            score = round(score.item(), 3)
            bbox = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + " " + str(score)
            txt_to_write += bbox + "\n"
        name = imageFile[:imageFile.rfind('.')]
        txt_writer = str(name) + '\n' + str(nums) + '\n' + txt_to_write
        file_name = name + '.txt'
        file_path = os.path.join(text_root, file_name)
        with open(file_path, 'w') as fw:
            fw.write(txt_writer)
