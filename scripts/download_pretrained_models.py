import argparse
import os
from os import path as osp

from basicsr.utils.download_util import load_file_from_url


def download_pretrained_models(method, file_urls):
    save_path_root = f'./weights'
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_url in file_urls.items():
        save_path = load_file_from_url(url=file_url, model_dir=save_path_root, progress=True, file_name=file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'method',
        type=str,
        help=("Options: 'LEDNet' 'CE-ZeroDCE'. Set to 'all' to download all the models."))
    args = parser.parse_args()

    file_urls = {
        'LEDNet': {
            'lednet.pth': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
            'lednet_retrain_500000.pth': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
            'lednetgan.pth': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth'
        },
        'CE-ZeroDCE': {
            'ce_zerodce.pth': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/ce_zerodce.pth'
        }
    }

    if args.method == 'all':
        for method in file_urls.keys():
            download_pretrained_models(method, file_urls[method])
    else:
        download_pretrained_models(args.method, file_urls[args.method])