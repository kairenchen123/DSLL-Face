import numpy as np
import os
import argparse
from tqdm import tqdm
import options
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loader import get_validation_data
import utils
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

def main():

    args = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
    utils.mkdir(args.result_dir)

    test_dataset = get_validation_data(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    model_restoration= utils.get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)

    print("===>Testing using weights: ", args.weights)
    utils.load_checkpoint(model_restoration,args.weights)
    model_restoration.cuda()
    model_restoration.eval()

    print("===>Testing mem using weights: ", args.mem_weights)
    model_restoration_memnet= utils.get_arch_mem(args)
    model_restoration_memnet = torch.nn.DataParallel(model_restoration_memnet)
    mem_checkpoint = torch.load(args.mem_weights)
    model_restoration_mem = model_restoration_memnet.module
    model_restoration_mem.key = mem_checkpoint['mem_key']
    model_restoration_mem.value = mem_checkpoint['mem_vale']
    model_restoration_mem.age = mem_checkpoint['mem_age']
    model_restoration_mem.top_index = mem_checkpoint['mem_index']

    model_restoration_mem.key = model_restoration_mem.key.cuda()
    model_restoration_mem.value = model_restoration_mem.value.cuda()
    model_restoration_mem.age = model_restoration_mem.age.cuda()
    model_restoration_mem.top_index = model_restoration_mem.top_index.cuda()
    model_restoration_mem.cuda()
    model_restoration_mem.eval()
    avg_pooling = nn.AdaptiveAvgPool2d((args.pooling_size,args.pooling_size))

    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []



        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_noisy = data_test[0].cuda()
            filenames = data_test[1]
            input_mem = data_test[3].cuda()

            # forward
            rgb_restored = model_restoration(rgb_noisy)

            # Ori: original output from image enhancer
            rgb_restored_ori = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
            rgb_restored_ori = (rgb_restored_ori*255.0).astype(np.uint8)

            filename,ext = os.path.splitext(filenames[0])

            # Mem: output after external memory
            # y_hat = torch.clamp(rgb_restored,0,1)
            # query = model_restoration_mem(input_mem)
            # top1_feature, _ = model_restoration_mem.topk_feature(query, args.testing_top_k)
            # top1_feature = top1_feature[:, 0, :].squeeze()
            # mean_out_mem = top1_feature
            # if args.pooling_mean == True:
            #     mean_out_mem = mean_out_mem.view(1,1,args.pooling_size,args.pooling_size)
            # else:
            #     mean_out_mem = mean_out_mem.view(1,3,args.pooling_size,args.pooling_size)
            # mean_out = avg_pooling(rgb_restored)
            #
            # if args.pooling_mean == True:
            #     mean_out = torch.mean(mean_out,1).unsqueeze(1)
            # ratio_map = (mean_out_mem/mean_out)
            # mean_cat = torch.cat((mean_out,mean_out_mem))
            # weight = torch.nn.functional.softmax(mean_cat,dim=0)
            # alpha1 = weight[0]
            # alpha2 = weight[1]
            # ratio_map_up = alpha1 + alpha2*ratio_map
            # restored_mem = torch.clamp(y_hat*ratio_map_up,0,1)
            #
            #
            # restored_mem = restored_mem.cpu().numpy().squeeze().transpose((1,2,0))
            # restored_mem = np.round(restored_mem*255.0).astype(np.uint8)




            if args.save_images:
                #ori
                subdir = 'ori'
                # path = f'{filename}_P{cur_psnr:>.4f}_S{cur_ssim:>.4f}{ext}'
                path = f'{filename}{ext}'
                utils.mkdir(os.path.join(args.result_dir,subdir))
                utils.save_img(os.path.join(args.result_dir,subdir,path), img_as_ubyte(rgb_restored_ori))


                #mem
                # subdir = 'mem'
                # # path = f'{filename}_P{cur_psnr_mem:>.4f}_S{cur_ssim_mem:>.4f}{ext}'
                # path = f'{filename}{ext}'
                # utils.mkdir(os.path.join(args.result_dir,subdir))
                # utils.save_img(os.path.join(args.result_dir,subdir,path), img_as_ubyte(restored_mem))



if __name__ == '__main__':
    main()


