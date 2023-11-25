from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import os
import cv2
from skimage import img_as_ubyte

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

pretrained = '/mnt/data2/ckr/Retinexformer/checkpoint/LOL_v2_synthetic.pth'
device = torch.device('cuda:0')
model_restoration = RetinexFormer(in_channels = 3, out_channels = 3, n_feat = 40, stage = 1, num_blocks = [1, 2, 2])
weight = torch.load(pretrained)
model_restoration.load_state_dict(weight['params'])
model_restoration.to(device)
model_restoration.eval()

inp_path = f'/mnt/data2/ckr/mmdetection/data/VOCdevkit/VOC2007/preJPEGImages/000005.png'
result_dir = '/mnt/data2/ckr/Retinexformer/output/LOL_v2_synthetic'
factor = 4
with torch.inference_mode():
    img = np.float32(load_img(inp_path)) / 255.

    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).cuda()

    # Padding in case images are not multiples of 4
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + factor) // factor) * \
           factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    restored = model_restoration(input_)

    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]

    restored = torch.clamp(restored, 0, 1).cpu(
    ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    save_img((os.path.join(result_dir, os.path.splitext(
        os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))