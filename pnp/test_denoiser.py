"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
# import multiprocessing
import pathlib
import random
import time
from collections import defaultdict

import numpy as np
import torch
from utils.fastmri.utils import generate_gro_mask

import matplotlib.pyplot as plt

# import bart
import sigpy as sp
import sigpy.mri as mr

from utils.fastmri import utils
from argparse import ArgumentParser
from utils.fastmri import tensor_to_complex_np
from eval import nmse, psnr

from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import SliceDataset

from utils.fastmri.models.PnP.train_denoiser_multicoil_brain import load_model
from utils import fastmri
import h5py
import scipy.misc
import PIL
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def psnr_2(gt, pred, zf=None):
    """ Compute Normalized Mean Squared Error (NMSE) """
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    plt.imshow(gt,cmap='gray')
    plt.savefig('other.png')
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def optimal_scale(target, recons, return_alpha=False):
    if recons.ndim == 3:
        alpha = np.sum(target * recons, axis=(1, 2), keepdims=True) / np.sum(recons ** 2, axis=(1, 2), keepdims=True)
    else:
        alpha = np.sum(target * recons, axis=(0, 1), keepdims=True) / np.sum(recons ** 2, axis=(0, 1), keepdims=True)
    # print(alpha)
    if return_alpha:
        return alpha * recons, alpha
    return alpha * recons


def get_gro_mask(mask_shape):
    # Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))


class DataTransform:
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, args, use_seed=None):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.use_seed = use_seed
        self.args = args
        self.mask = None
        if args.mask_path is not None:
            self.mask = torch.load(args.mask_path)

    def __call__(self, kspace, mk, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array, optional): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same shape as kspace.
                fname (str): File name containing the current data item
                slice (int): The index of the current slice in the volume
        """
        kspace = transforms.to_tensor(kspace)
        # sens = mr.app.EspiritCalib(tensor_to_complex_np(kspace)).run()
        sens = np.ones((320, 320, 2))
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask) + 0.0

        target = fastmri.complex_abs(fastmri.ifft2c(kspace))

        return masked_kspace, mask, sens, target, fname, slice



def denoiser(noisy, model, args):
    # add rotate
    noisy, rot_angle = transforms.best_rotate(noisy, args.rotation_angles)

    # normalize
    if (args.normalize == 'max') or (args.normalize == 'std'):
        noisy, scale = transforms.denoiser_normalize(noisy, is_complex=True, use_std=args.normalize == 'std')
    elif args.normalize == 'constant':
        print('in here')
        scale = 0.0016
        noisy = noisy * (1 / scale)
    else:
        scale = 1

    if args.denoiser_mode == 'mag':
        mag = transforms.complex_abs(noisy)
        phase = transforms.phase(noisy)
        denoised_mag = model(mag[None, None, ...])
        denoised_mag = denoised_mag[0, 0, ...]
        denoised_image = transforms.polar_to_rect(denoised_mag, phase)

    elif args.denoiser_mode == '2-chan':
        # move real/imag to channel position
        noisy = noisy.permute(2, 0, 1).unsqueeze(0)
        denoised_image = model(noisy).squeeze(0).permute(1, 2, 0)

    elif args.denoiser_mode == 'real-imag':
        # move real/imag to batch position
        noisy = noisy.permute(2, 0, 1).unsqueeze(1)
        denoised_image = model(noisy).squeeze(1).permute(1, 2, 0)

    # unnormalize
    denoised_image = denoised_image * scale

    # undo rotate
    denoised_image = transforms.polar_to_rect(transforms.complex_abs(denoised_image),
                                              transforms.phase(denoised_image) - rot_angle)

    return denoised_image


def find_spec_rad(mri, steps, x):
    # init x
    x = torch.randn_like(x)
    x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2))

    # power iteration
    for i in range(steps):
        x = fastmri.ifft2c(fastmri.fft2c(x))
        spec_rad = torch.sqrt(torch.sum(torch.abs(x) ** 2))
        x = x / spec_rad

    return spec_rad


def main(args):
    # with multiprocessing.Pool(20) as pool:
    #     start_time = time.perf_counter()
    #     outputs = pool.map(run_model, range(len(data)))
    #     time_taken = time.perf_counter() - start_time
    #     logging.info(f'Run Time = {time_taken:}s')
    #     save_outputs(outputs, args.output_path)

    # handle pytorch device
    device = torch.device("cuda")

    # load model
    if args.checkpoint is not None:
        if args.natural_image:
            model = torch.load(args.checkpoint)
        else:
            _, model, _ = load_model(args.checkpoint)
        model.to(device)
        model.eval()
    else:
        model = None

    with h5py.File('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_train/file_brain_AXT2_200_2000006.h5', 'r') as hf:
        kspace = transforms.to_tensor(hf['kspace'][()])[0]
        zf_kspace = (kspace * get_gro_mask(kspace.shape)) + 0.0
        complex_image = fastmri.ifft2c(kspace)
        zfr = fastmri.ifft2c(zf_kspace)
        print(f'TEST PSNR: {psnr_2(fastmri.complex_abs(complex_image),fastmri.complex_abs(zfr))}')
        noisy = complex_image + 0.2 * torch.randn(complex_image.size())
        denoised = denoiser(noisy, model, args)
        print(f'PRE PSNR: {psnr_2(fastmri.complex_abs(complex_image),fastmri.complex_abs(noisy))}\nPOST PSNR: {psnr_2(fastmri.complex_abs(complex_image),fastmri.complex_abs(denoised).detach())}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path,
                        default='/storage/fastMRI_brain/data/Matt_preprocessed_data/T2')
    parser.add_argument('--output-path', type=pathlib.Path, default='out',
                        help='Path to save the reconstructions to')
    parser.add_argument('--snr', type=float, default=None, help='measurement noise')
    parser.add_argument('--project', default=False, action='store_true',
                        help='replace loss prox with projection operator')
    parser.add_argument('--algorithm', type=str, default='pnp-admm',
                        help='Algorithm used (pnp-pg, pnp-admm, red-admm, red-gd)')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--inner-iters', type=int, default=3,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--inner-denoiser-iters', type=int, default=1,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--step-size', type=float, default=None,
                        help='Step size parameter')
    parser.add_argument('--lamda', type=float, default=0.01, help='Regularization weight parameter')
    parser.add_argument('--relaxation', type=float, default=0.000, help='Relaxation of denoiser in PnP-PG')
    parser.add_argument('--beta', type=float, default=0.001, help='ADMM Penalty parameter')
    parser.add_argument('--device', type=int, default=0, help='Cuda device (-1 for CPU)')
    parser.add_argument('--denoiser-mode', type=str, default='2-chan', help='Denoiser mode (mag, real_imag, 2-chan)')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP/best_model.pt',
                        help='Path to an existing checkpoint.')
    parser.add_argument("--debug", default=True, action="store_true", help="Debug mode")
    parser.add_argument("--test-idx", type=int, default=1, help="test index image for debug mode")
    parser.add_argument("--natural-image", default=False, action="store_true",
                        help="Uses a pretrained DnCNN rather than a custom trained network")
    parser.add_argument("--normalize", type=str, default=None,
                        help="Type of normalization going into denoiser (None, 'max', 'std')")
    parser.add_argument('--rotation-angles', type=int, default=0,
                        help='number of rotation angles to try (<1 gives no rotation)')
    parser.add_argument("--accel", default=False, action='store_true', help="apply nesterov acceleration")
    parser.add_argument("--use-mid-slices", default=False, action='store_true', help="use only middle slices")
    parser.add_argument("--scanner-strength", type=float, default=None,
                        help="Leave as None for all, >2.2 for 3, > 2.2 for 1.5")
    parser.add_argument('--mask-path', type=str, default=None, help='Path to mask (saved as Tensor)')
    parser.add_argument('--nc', type=int, default=4, help='number of coils to simulate')
    parser.add_argument('--coil-root', type=str, default='/home/reehorst.3/Documents/Reehorst_coil_maps/',
                        help='path to coil directory')
    parser.add_argument("--scanner-mode", type=str, default=None,
                        help="Leave as None for all, other options are PD, PDFS")
    parser.add_argument("--espirit-cal", default=False, action="store_true", help="set to use espririt calibrated maps")
    parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
    parser.add_argument('--rss-target', default=False, action='store_true', help="Use rss as target (otherwise use gt)")
    parser.add_argument('--optimal-scaling', default=False, action='store_true', help="Optimal scaling")
    args = parser.parse_args()

    # restrict visible cuda devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    main(args)
