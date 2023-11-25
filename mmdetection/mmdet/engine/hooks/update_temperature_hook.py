from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from typing import Optional
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet.registry import HOOKS
from mmdet.apis.inference import init_detector


@HOOKS.register_module()
class UpdateTemperature(Hook):
    def after_train_epoch(self, runner: Runner):
        # config_file = '/home/gzhu2023/gzhu2023/ckr/mmdetection/local_config/dino/dino-4scale_r50_8xb2-12e_coco.py'
        # model = init_detector(config_file)
        model = runner.model
        epoch = runner.epoch
        if epoch < 10:
            backbone = model.backbone
            backbone.update_temperature()