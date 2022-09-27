import cv2
import random
import numpy as np
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.archs.zerodce_arch import ConditionZeroDCE
from basicsr.data.degradations import random_add_gaussian_noise, random_add_poisson_noise

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import DATASET_REGISTRY

class RandomLowLight(object):
    def __init__(self, low_light_net, exp_ranges=[0.05, 0.3]):
        self.threshold = 0.97
        self.exp_range = exp_ranges
        # self.exp_range = [0.1, 0.3]
        self.low_light_net = low_light_net

    def __call__(self, img):
        # img range: [0, 1], float32.
        exp_degree = random.uniform(*self.exp_range)
        h,w,_ = img.shape
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        # 0<=L<=100, -127<=a<=127, -127<=b<=127
        l_channel_t = torch.from_numpy(l_channel).view(1,1,h,w).cuda()
        l_channel_f = l_channel_t/100.0
        exp_map = exp_degree * torch.ones_like(l_channel_f)

        stuated_map = (l_channel_f>self.threshold).int()
        exp_map = exp_map*(1-stuated_map) + l_channel_f*stuated_map

        low_light_l = (self.low_light_net(l_channel_f, exp_map)*100).squeeze().cpu().detach().numpy()

        # a_channel = a_channel*(low_light_l/(l_channel+1e-8))
        # b_channel = b_channel*(low_light_l/(l_channel+1e-8))
        # low_light_img = np.dstack((low_light_l, a_channel,b_channel))
        # low_light_img = cv2.cvtColor(low_light_img, cv2.COLOR_LAB2BGR)*255

        scale = low_light_l/(l_channel+1e-8)
        scale = np.dstack([scale]*3)
        low_light_img = img*scale

        return low_light_img

class AddGaussianNoise(object):
    def __init__(self):
        self.noise_range = [0, 8]
        self.poisson_scale_range = [0.1, 1.2]
    def __call__(self, img):
        if np.random.uniform() < 0.5:
            img = random_add_gaussian_noise(img, sigma_range=self.noise_range, gray_prob=0.3)
        else:
            img = random_add_poisson_noise(img, scale_range=self.poisson_scale_range, gray_prob=0.3)
        return img


@DATASET_REGISTRY.register()
class LOLImageDataset(data.Dataset):
    """Low-light image dataset for low-light image enhancement.

    Read GT image and generate low-light image on the fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(LOLImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.generate_lol_img = opt.get('generate_lol_img', True)
        self.add_gaussian_noise = opt.get('add_noise', True)
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.crop_size = opt.get('crop_size', 256)
        self.scale = opt.get('scale', 1)

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder, self.lq_folder = opt.get('dataroot_gt', None), opt.get('dataroot_lq', None)
        if self.generate_lol_img:
            # load pretrained model
            ckpt_path = load_file_from_url(
                'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/ce_zerodce.pth', 
                model_dir='./weights', progress=True, file_name=None)
            low_light_net = ConditionZeroDCE().cuda()
            low_light_net.load_state_dict(torch.load(ckpt_path))
            low_light_net.eval()
            self.lol_generator = RandomLowLight(low_light_net, exp_ranges=[0.05, 0.3])
            self.paths = paths_from_folder(self.gt_folder, recursive=True, full_path=True)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        if self.add_gaussian_noise:
            self.noise_adder = AddGaussianNoise()
        if self.opt['phase'] == 'train':
            random.shuffle(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        if self.generate_lol_img:
            gt_path = self.paths[index]
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
        else:
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            if self.generate_lol_img:
                img_gt, img_lq = paired_random_crop(img_gt, img_gt, self.crop_size, self.scale, gt_path)
                # random low-light simulation
                img_lq = self.lol_generator(img_lq)
            else:
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.crop_size, self.scale, gt_path)
            
            # random noise
            if self.add_gaussian_noise:
                img_lq = self.noise_adder(img_lq)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.use_flip, self.use_rot)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        normalize(img_lq, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.paths)
