"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
# import multiprocessing
import pathlib
import random
import time
from collections import defaultdict
import pickle

import numpy as np
import torch

import matplotlib

matplotlib.use('TKAgg')

import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

from utils.fastmri import utils
from argparse import ArgumentParser
from utils.fastmri.data.subsample import MaskFunc
from utils.fastmri import tensor_to_complex_np
from utils.fastmri.evaluate import nmse
from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import SliceDataset

from utils.fastmri.models.PnP.dncnn import DnCNN
from utils.fastmri.models.PnP.train_dncnn import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
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

    def __init__(self, mask_func):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.mask_func = mask_func

    def __call__(self, kspace, target, attrs, fname, slice):
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
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask) + 0.0
        return masked_kspace, mask, target, fname, slice


def create_data_loader(args):
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceDataset(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(dev_mask),
        challenge=args.challenge,
        sample_rate=args.sample_rate
    )
    return data


def denoiser(noisy, model, mag_only=False):
    if mag_only:
        mag = transforms.complex_abs(noisy)
        phase = transforms.phase(noisy)
        mag, scale = transforms.denoiser_normalize(mag)
        denoised_mag = torch.clamp(model(mag[None, None, ...]), 0, 1)
        denoised_mag = denoised_mag[0, 0, ...]
        denoised_mag = transforms.denoiser_denormalize(denoised_mag, scale)
        return transforms.polar_to_rect(denoised_mag, phase)

    else:
        # normalize
        image, scale = transforms.denoiser_normalize(noisy, is_complex=True, use_std=False)
        # move real/imag to channel position
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.clamp(-1, 1)  # should only clamp if using std_normalization
        denoised_image = (model(image) * scale).squeeze(0).permute(1, 2, 0)
        return denoised_image


def pnp_pg(y, mri, model, args):
    with torch.no_grad():
        x = mri.H(y)
        for k in range(args.num_iters):
            # perform gradient step
            z = x - args.step_size * mri.H(mri.A(x) - y)
            # perform denoising
            x = denoiser(z, model, args.mag_only)
        return x


def red_gd(y, mri, model, args):
    with torch.no_grad():
        x = mri.H(y)
        for k in range(args.num_iters):
            # perform denoising
            fx = denoiser(x, model, args.mag_only)
            # perform gradient step
            x = x - args.step_size * (mri.H(mri.A(x) - y) + args.lamda * (x - fx))
        return x


class A_mri:
    def __init__(self, sens_maps, mask):
        self.sens_maps = sens_maps
        self.mask = mask.unsqueeze(3)

    def A(self, x):
        y = transforms.complex_mult(x, self.sens_maps)
        y_fft = transforms.fft2(y)
        out = self.mask * y_fft
        return out

    def H(self, x):
        y = self.mask * x
        y_ifft = transforms.ifft2(y)
        out = torch.sum(transforms.complex_mult(y_ifft, transforms.complex_conj(self.sens_maps)), dim=0)
        return out


# class A_mri:
#     def __init__(self, sens_maps, mask):
#         A = mr.linop.Sense(sens_maps, weights=mask)
#         self.A = sp.to_pytorch_function(A, input_iscomplex=True, output_iscomplex=True)
#         self.H = sp.to_pytorch_function(A.H, input_iscomplex=True, output_iscomplex=True)


def cs_pnp(args, model, kspace, mask):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """

    if args.challenge == 'singlecoil':
        kspace = kspace.unsqueeze(0)
    mask = mask.permute(0, 2, 1)
    kspace_np = tensor_to_complex_np(kspace)
    mask = mask.cpu().numpy()

    device = sp.Device(args.device)

    # Estimate sensitivity maps
    # sens_maps = bart.bart(1, f'ecalib -d0 -m1', kspace)
    sens_maps = mr.app.EspiritCalib(kspace_np, device=device).run()

    # handle pytorch device
    device = torch.device("cuda:{0:d}".format(args.device) if torch.cuda.is_available() else "cpu")
    sens_maps = transforms.to_tensor(sens_maps.astype('complex64')).to(device)
    mask = transforms.to_tensor(mask).to(device)

    mri = A_mri(sens_maps, mask)

    # Use PnP-PG to reconstruct the image
    kspace = kspace.to(device)

    if args.algorithm == 'pnp-pg':
        pred = pnp_pg(kspace, mri, model, args)

    elif args.algorithm == 'red-gd':
        pred = red_gd(kspace, mri, model, args)

    pred = transforms.complex_abs(pred).cpu().numpy()
    # Crop the predicted image to the correct size
    return transforms.center_crop(pred, (args.resolution, args.resolution))


# def run_model(i):
#     masked_kspace, mask, fname, slice = data[i]
#     prediction = cs_pnp(args, masked_kspace, mask)
#     return fname, slice, prediction

def main(args):
    # with multiprocessing.Pool(20) as pool:
    #     start_time = time.perf_counter()
    #     outputs = pool.map(run_model, range(len(data)))
    #     time_taken = time.perf_counter() - start_time
    #     logging.info(f'Run Time = {time_taken:}s')
    #     save_outputs(outputs, args.output_path)

    # handle pytorch device
    device = torch.device("cuda:{0:d}".format(args.device) if torch.cuda.is_available() else "cpu")

    # load model
    _, model, _ = load_model(args.checkpoint)
    model.to(device)
    model.eval()

    # non pooled version
    start_time = time.perf_counter()
    outputs = []

    if args.debug:
        test_array = [20]
    else:
        test_array = range(len(data))
    
    slize_time = 0
    for i in test_array:
        print('Test ' + str(i) + ' of ' + str(len(test_array)), end='\r')
        masked_kspace, mask, target, fname, slice = data[i]
        slice_time = time.perf_counter()
        prediction = cs_pnp(args, model, masked_kspace, mask)
        slice_time = time.perf_counter() - slice_time
        if args.debug:
            display = np.concatenate([prediction, target], axis=1)
            print('NMSE = ' + str(nmse(target, prediction)))
            pl.ImagePlot(display)
        outputs.append([fname, slice, prediction, slice_time])
    time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    save_outputs(outputs, args.output_path)


def save_outputs(outputs, output_path):
    reconstructions = defaultdict(list)
    times = defaultdict(list)

    for fname, slice, pred, recon_time in outputs:
        reconstructions[fname].append((slice, pred))
        times[fname].append((slice, recon_time))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    utils.save_reconstructions(reconstructions, output_path)
    with open('out/pnp_times.pkl', 'wb') as f:
        pickle.dump(times, f)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--output-path', type=pathlib.Path, default=pathlib.Path(out),
                        help='Path to save the reconstructions to')
    parser.add_argument('--algorithm', type=str, default='red-gd',
                        help='Algorithm used (pnp-pg, pnp-admm, red-admm, red-gd)')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--step-size', type=float, default=0.01,
                        help='Step size parameter')
    parser.add_argument('--lamda', type=float, default=0.01, help='Regularization weight parameter')
    parser.add_argument('--beta', type=float, default=0.001, help='ADMM Penalty parameter')
    parser.add_argument('--device', type=int, default=-1, help='Cuda device (-1 for CPU)')
    # parser.add_argument('--denoiser-mode', type=str, default='mag', help='Denoiser mode (mag, real_imag)')
    parser.add_argument('--checkpoint', type=str, default='/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP/checkpoints/best_model.pt',
                        help='Path to an existing checkpoint.')
    parser.add_argument("--mag-only", default=False, action="store_true", help="Magnitude only denoising")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = create_data_loader(args)
    main(args)
