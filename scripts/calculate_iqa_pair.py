import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
import glob
from basicsr.utils import scandir

import pyiqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='[path_to_results]')
    parser.add_argument('--gt_path', type=str, default='[path_to_gt]')
    parser.add_argument('--metrics', nargs='+', default=['psnr','ssim','lpips'])
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces')

    args = parser.parse_args()

    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    if args.gt_path.endswith('/'):  # solve when path ends with /
        args.gt_path = args.gt_path[:-1]

    # Initialize metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iqa_psnr, iqa_ssim, iqa_lpips = None, None, None
    score_psnr_all, score_ssim_all, score_lpips_all = [], [], []
    print(args.metrics)
    if 'psnr' in args.metrics:
      iqa_psnr = pyiqa.create_metric('psnr').to(device)
      iqa_psnr.eval()
    if 'ssim' in args.metrics:
      iqa_ssim = pyiqa.create_metric('ssim').to(device)
      iqa_ssim.eval()
    if 'lpips' in args.metrics:
      # iqa_lpips = pyiqa.create_metric('lpips').to(device)
      iqa_lpips = pyiqa.create_metric('lpips-vgg').to(device)
      iqa_lpips.eval() 

    img_out_paths = sorted(list(scandir(args.result_path, suffix=('jpg', 'png'), 
                                    recursive=True, full_path=True)))
    total_num = len(img_out_paths)

    for i, img_out_path in enumerate(img_out_paths):
        img_name = img_out_path.replace(args.result_path+'/', '')
        cur_i = i + 1
        print(f'[{cur_i}/{total_num}] Processing: {img_name}')
        img_out = cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()
        try:
          img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
          img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
          img_gt = np.transpose(img_gt, (2, 0, 1))
          img_gt = torch.from_numpy(img_gt).float()
          with torch.no_grad():
            img_out = img_out.unsqueeze(0).to(device)
            img_gt = img_gt.unsqueeze(0).to(device)
            if iqa_psnr is not None:
              score_psnr_all.append(iqa_psnr(img_out, img_gt).item())
            if iqa_ssim is not None:
              score_ssim_all.append(iqa_ssim(img_out, img_gt).item()) 
            if iqa_lpips is not None:
              score_lpips_all.append(iqa_lpips(img_out, img_gt).item())
        except:
          print(f'skip: {img_name}')
          continue
        if (i+1)%20 == 0:
          print(f'[{cur_i}/{total_num}] PSNR: {sum(score_psnr_all)/len(score_psnr_all)}, \
                  SSIM: {sum(score_ssim_all)/len(score_ssim_all)}, \
                  LPIPS: {sum(score_lpips_all)/len(score_lpips_all)}\n')

    print('-------------------Final Scores-------------------\n')
    print(f'PSNR: {sum(score_psnr_all)/len(score_psnr_all)}, \
            SSIM: {sum(score_ssim_all)/len(score_ssim_all)}, \
            LPIPS: {sum(score_lpips_all)/len(score_lpips_all)}')
