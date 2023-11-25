from RetinexFormer_arch import RetinexFormer
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import cv2

pretrained = '/mnt/data2/ckr/Retinexformer/checkpoint/LOL_v2_real.pth'
device = torch.device('cuda:2')
model_restoration = RetinexFormer(in_channels = 3, out_channels = 3, n_feat = 40, stage = 1, num_blocks = [1, 2, 2])
weight = torch.load(pretrained)
model_restoration.load_state_dict(weight['params'])
model_restoration.to(device)
model_restoration.eval()


image_path = f'/mnt/data2/ckr/mmdetection/data/track1.2_test_sample/0.png'
image_name = os.path.basename(image_path)

image = cv2.imread(image_path)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640))
])
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)
output = model_restoration(image)
image = output.cpu().clone()
out = torch.squeeze(output)
to_pil = transforms.ToPILImage()
out = to_pil(out)
out_np = np.array(out)
cv2.imwrite(f'/mnt/data2/ckr/mmdetection/work_dirs/darkface_output_{image_name}',out_np)