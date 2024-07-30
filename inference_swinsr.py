import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


from LightingFormer_arch import LightingFormer as net

from torch.nn import functional as F
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=r"G:\***.pth")
    parser.add_argument('--folder_lq', type=str, default=r"G:\eval15\low", help='input low-quality test image folder')
    parser.add_argument('--save_dir', type=str, default=r".\res", help='input ground-truth test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        print(f'loading error')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    folder, border, window_size = args.folder_lq, 0, 32

    save_dir=args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        print(imgname)
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            mod_pad_h, mod_pad_w = 0, 0
            _, _, h, w = img_lq.size()
            if h % window_size != 0:
                mod_pad_h = window_size - h % window_size
            if w % window_size != 0:
                mod_pad_w = window_size - w % window_size
            img_lq = F.pad(img_lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            output = model(img_lq)
            output = output[:, :, :h_old, :w_old]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}', output)




def get_bare_model(net):
    """Get bare model, especially under wrapping with
    DistributedDataParallel or DataParallel.
    """
    if isinstance(net, (DataParallel, DistributedDataParallel)):
        net = net.module
    return net


def define_model(args):
    model = net()
    param_key_g = 'params_ema'
    pretrained_model = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)

    return model




def get_image_pair(args, path):
    a=os.path.basename(path)
    img_gt = None
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    return a, img_lq, img_gt


if __name__ == '__main__':
    main()
