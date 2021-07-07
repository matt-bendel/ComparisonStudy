"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing
import pathlib
import time
from argparse import ArgumentParser
from collections import defaultdict

from utils.fastmri import save_reconstructions
import numpy as np
import torch

from utils.fastmri import tensor_to_complex_np
from utils.fastmri.data import SliceDataset
from utils.fastmri.data import transforms as T
from utils.fastmri.data.subsample import MaskFunc
import sigpy.mri as mr
import random

from utils.fastmri.utils import generate_gro_mask
import psutil

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

class DataTransform(object):
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

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        kspace = T.to_tensor(kspace)
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask + 0.0)

        return (masked_kspace, mask, target, fname, slice)

def create_data_loader(args):
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceDataset(
        root=args.data_path / f'singlecoil_val',
        transform=DataTransform(dev_mask),
        challenge='singlecoil',
        sample_rate=args.sample_rate
    )
    return data

def cs_total_variation(args, kspace, mask, slice):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
    based reconstruction algorithm using the BART toolkit.

    Args:
        args (argparse.Namespace): Arguments including ESPIRiT parameters.
        reg_wt (float): Regularization parameter.
        crop_size (tuple): Size to crop final image to.

    Returns:
        np.array: Reconstructed image.
    """
    crop_size = (320, 320)

    kspace = kspace.unsqueeze(0)
    mask = mask.permute(0,2,1)
    kspace = tensor_to_complex_np(kspace)
    mask = mask.cpu().numpy()

    # device = sp.Device(args.device)
    # Estimate sensitivity maps
    if args.challenge == 'singlecoil':
        sens_maps = np.ones(kspace.shape)
    else:
        sens_maps = mr.app.EspiritCalib(kspace).run()

    # use Total Variation Minimization to reconstruct the image
    pred = mr.app.TotalVariationRecon(kspace, sens_maps, 1, max_iter=args.num_iters).run()
    pred = torch.from_numpy(np.abs(pred))

    cropped_image = T.center_crop(pred, crop_size)

    return cropped_image


def save_outputs(outputs, output_path):
    """Saves reconstruction outputs to output_path."""
    reconstructions = defaultdict(list)
    for fname, slice_num, pred, recon_time in outputs:
        reconstructions[fname].append((slice_num, pred))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    save_reconstructions(reconstructions, output_path)


def run_model(idx):
    """
    Run BART on idx index from dataset.

    Args:
        idx (int): The index of the dataset.

    Returns:
        tuple: tuple with
            fname: Filename
            slice_num: Slice number.
            prediction: Reconstructed image.
    """
    start_time = time.perf_counter()
    masked_kspace, mask, target, fname, slice_num = data[idx]

    prediction = cs_total_variation(
        args, masked_kspace, mask, slice_num
    )

    recon_time = time.perf_counter() - start_time
    return fname, slice_num, prediction, recon_time


def run_cs(args):
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(data)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(data)))
            time_taken = time.perf_counter() - start_time

    print(f"Run Time = {time_taken} s")
    save_outputs(outputs, args.output_path)


def create_arg_parser():
    parser = ArgumentParser(add_help=False)

    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        help="Which challenge",
        default='singlecoil'
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Percent of data to run",
    )
    parser.add_argument(
        "--mask_type", choices=["random", "equispaced"], default="random", type=str
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test_dir", "challenge"],
        default="val",
        type=str,
    )
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int)
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float)

    return parser

parser = create_arg_parser()
parser.add_argument('--output_path', type=pathlib.Path, default=pathlib.Path('out'),
                    help='Path to save the reconstructions to')
parser.add_argument('--num-iters', type=int, default=200,
                    help='Number of iterations to run the reconstruction algorithm')
parser.add_argument('--reg-wt', type=float, default=1e-3,
                    help='Regularization weight parameter')
parser.add_argument('--num-procs', type=int, default=4,
                    help='Number of processes. Set to 0 to disable multiprocessing.')
parser.add_argument('--device', type=int, default=0,
                    help='Cuda device idx (-1 for CPU)')
parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

data = create_data_loader(args)
if __name__ == "__main__":
    run_cs(args)