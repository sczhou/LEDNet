import os
import cv2
import argparse
import glob
import torch
import random
import numpy as np
from basicsr.utils import imwrite, scandir
from basicsr.utils.download_util import load_file_from_url

from basicsr.archs.zerodce_arch import ConditionZeroDCE


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default='/mnt/lustre/sczhou/datasets/deblurring/LOLBlur/high_sharp_scaled')
    parser.add_argument('--result_path', type=str, default='results/darken_imgs')
    parser.add_argument('--model_path', type=str, default='weights/ce_zerodce.pth',)

    args = parser.parse_args()
    # load pretrained model
    ckpt_path = load_file_from_url(
        'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/ce_zerodce.pth', 
        model_dir='./weights', progress=True, file_name=None)

    # ------------------------ Darkness range ------------------------
    threshold = 0.97 # threshold for saturated regions
    exp_range = [0.05, 0.3]
    # exp_range = [0.1, 0.3]

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]
    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'

    # ------------------ set up EC-ZeroDCE network -------------------
    net = ConditionZeroDCE().to(device)
    ckpt_path = args.model_path
    # ckpt_path = load_file_from_url(url='xx', model_dir='./weights', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    img_paths = sorted(list(scandir(args.test_path, suffix=('jpg', 'png'), recursive=True, full_path=True)))

    for img_path in img_paths:
        # img_name = os.path.basename(img_path)
        img_name = img_path.replace(args.test_path+'/', '')
        print(f'Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # get random darkness
        exp_degree = random.uniform(*exp_range)

        # inference
        img = img.astype('float32')/255.0
        h,w,_ = img.shape
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        # 0<=L<=100, -127<=a<=127, -127<=b<=127
        l_channel_t = torch.from_numpy(l_channel).view(1,1,h,w).cuda()
        l_channel_f = l_channel_t/100.0
        exp_map = exp_degree * torch.ones_like(l_channel_f)
        stuated_map = (l_channel_f>threshold).int()
        exp_map = exp_map*(1-stuated_map) + l_channel_f*stuated_map

        with torch.no_grad():
          low_light_l = (net(l_channel_f, exp_map)*100).squeeze().cpu().detach().numpy()
        torch.cuda.empty_cache()

        # a_channel = a_channel*(low_light_l/(l_channel+1e-8))
        # b_channel = b_channel*(low_light_l/(l_channel+1e-8))
        # low_light_img = np.dstack((low_light_l, a_channel,b_channel))
        # low_light_img = cv2.cvtColor(low_light_img, cv2.COLOR_LAB2RGB)*255

        scale = low_light_l/(l_channel+1e-8)
        scale = np.dstack([scale]*3)
        low_light_img = img*scale*255

        img_out = low_light_img.clip(0,255).astype('uint8')

        # save darken img
        save_restore_path = img_path.replace(args.test_path, result_root)
        imwrite(img_out, save_restore_path)

    print(f'\nAll results are saved in {result_root}')
