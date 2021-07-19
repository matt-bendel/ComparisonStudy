"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import sys
from collections import defaultdict
import os
import time
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import fastmri
from utils.fastmri.data.mri_data import SliceDataset
from argparse import ArgumentParser
from utils.fastmri import save_reconstructions
from eval import nmse, psnr

from utils.fastmri.data import transforms
from utils.fastmri.models.unet.unet import UnetModel


from utils.fastmri.utils import generate_gro_mask

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
    Data Transformer for running U-Net models on a test dataset.
    """

    def __init__(self, args, use_seed=False):
        """
        Args:
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.use_seed = use_seed
        self.args = args
        self.mask = None

    def __call__(self, kspace, mk, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.Array): k-space measurements
            target (numpy.Array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object
            fname (pathlib.Path): Path to the input file
            slice (int): Serial number of the slice
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Normalized zero-filled input image
                mean (float): Mean of the zero-filled image
                std (float): Standard deviation of the zero-filled image
                fname (pathlib.Path): Path to the input file
                slice (int): Serial number of the slice
        """
        kspace = transforms.to_tensor(kspace)
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask) + 0.0

        # Inverse Fourier Transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # Absolute value
        image = fastmri.complex_abs(image)

        # Normalize input
        image, mean, std = transforms.normalize_instance(image)
        image = image.clamp(-6, 6)

        if self.args.data_split == 'val':
            target = transforms.to_tensor(target)
            # Normalize target
            target = transforms.normalize(target, mean, std, eps=1e-11)
            # target = target.clamp(-6, 6)
            return image, mean, std, fname, slice, target

        return image, mean, std, fname, slice


def create_data_loaders(args):
    data = SliceDataset(
        root=args.data_path / f'singlecoil_val',
        transform=DataTransform(args),
        sample_rate=1.0,
        challenge='singlecoil',
    )

    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=64,
        pin_memory=True,
    )
    return data_loader


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    args = checkpoint['args']
    model = UnetModel(1, 1, args.num_chans, args.num_pools, args.drop_prob).to(torch.device('cuda'))
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model


def run_unet(args, model, data_loader):
    model.eval()
    a_metrics = []

    reconstructions = defaultdict(list)
    times = defaultdict(list)

    with torch.no_grad():
        if args.debug:
            input, mean, std, fname, slice = data_loader[0:5]  # data_loader is actually a tensor
            input = input.unsqueeze(0).unsqueeze(0).to(args.device)
            recon = model(input).to('cpu').squeeze()
            recon = recon * std + mean
            recon = recon.cpu().numpy()
            display = recon
            target = data_loader[5]
            target = target * std + mean
            target = target.cpu().numpy()
            # Compute metrics
            NMSE = nmse(target, recon)
            rSNR = 10 * np.log10(1 / NMSE)
            PSNR = psnr(target, recon)
            print(f'NMSE: {NMSE}')
            print(f'PSNR: {PSNR}')
            print(f'rSNR: {rSNR}')

            return
        for data in data_loader:
            input, mean, std, fnames, slices = data[0:5]
            print(input.shape)
            input = input.unsqueeze(1).to(args.device)
            start_time = time.perf_counter()
            recons = model(input).to('cpu').squeeze(1)
            total = time.perf_counter() - start_time
            avg_time = total / recons.shape[0]
            targets = data[5]
            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))
                times[fnames[i]].append(avg_time)
                targets[i] = targets[i] * std[i] + mean[i]

            targets = targets.cpu().numpy()
            recons = recons.cpu().numpy()
            for i in range(recons.shape[0]):
                # compute metrics
                NMSE = nmse(targets[i], recons[i])
                rSNR = 10 * np.log10(1 / NMSE)
                PSNR = psnr(targets[i], recons[i])
                metrics = [NMSE, PSNR, rSNR]
                a_metrics.append(metrics)

    if args.data_split == 'val':
        # Print metrics
        a_metrics = np.array(a_metrics)
        a_names = ['NMSE', 'PSNR', 'SSIM', 'rSNR']
        mean_metrics = np.mean(a_metrics, axis=0)
        std_metrics = np.std(a_metrics, axis=0)
        for i in range(len(a_names)):
            print(a_names[i] + ': ' + str(mean_metrics[i]) + ' +/- ' + str(2 * std_metrics[i]))

        if args.out_dir is not None:
            args.out_dir.mkdir(exist_ok=True)
            metric_file = args.out_dir / 'metrics.np'
            np.save(metric_file, a_metrics)

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    with open('out/pnp_times.pkl', 'wb') as f:
        pickle.dump(times, f)
    return reconstructions


def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint).to(args.device)
    start_time = time.perf_counter()
    reconstructions = run_unet(args, model, data_loader)
    time_taken = time.perf_counter() - start_time
    print(f'Run Time = {time_taken:}s')
    if args.out_dir is not None:
        save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="Path to the data",
    )
    # parser.add_argument('--mask-kspace', action='store_true',
    #                     help='Whether to apply a mask (set to True for val data and False '
    #                          'for test data')
    parser.add_argument('--data-split', choices=['val', 'test'], default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, default='out',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=int, default=1, help='Which cuda device to run on (give idx as int)')
    parser.add_argument('--snr', type=float, default=None, help='measurement noise')
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    parser.add_argument("--test-idx", type=int, default=0, help="test index image for debug mode")
    parser.add_argument("--use-mid-slices", default=False, action='store_true', help="use only middle slices")
    parser.add_argument("--scanner-strength", type=float, default=None,
                        help="Leave as None for all, >2.2 for 3, > 2.2 for 1.5")
    parser.add_argument("--scanner-mode", type=str, default=None,
                        help="Leave as None for all, other options are PD, PDFS")
    parser.add_argument('--mask-path', type=str, default=None, help='Path to mask (saved as Tensor)')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    # restrict visible cuda devices
    if args.device >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    main(args)
