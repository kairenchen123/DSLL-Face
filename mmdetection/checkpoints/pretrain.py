import torch

num_classes = 2
model_coco = torch.load("./yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth")

# for key, value in model_coco["state_dict"].items():
# print(key)

# # weight
model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.0.fc_cls.weight"][:num_classes,
                                                                 :]
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.1.fc_cls.weight"][:num_classes,
                                                                 :]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.weight"] = model_coco["state_dict"][
                                                                     "roi_head.bbox_head.2.fc_cls.weight"][:num_classes,
                                                                 :]
# bias
model_coco["state_dict"]["roi_head.bbox_head.0.fc_cls.bias"] = model_coco["state_dict"][
                                                                   "roi_head.bbox_head.0.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["roi_head.bbox_head.1.fc_cls.bias"] = model_coco["state_dict"][
                                                                   "roi_head.bbox_head.1.fc_cls.bias"][:num_classes]
model_coco["state_dict"]["roi_head.bbox_head.2.fc_cls.bias"] = model_coco["state_dict"][
                                                                   "roi_head.bbox_head.2.fc_cls.bias"][:num_classes]
# save new model
torch.save(model_coco, "./yolov3_d53_mstrain-608_273e_coco_classes_%d.pth" % num_classes)