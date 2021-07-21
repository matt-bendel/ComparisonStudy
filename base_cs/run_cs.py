"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import multiprocessing
import pickle

import pathlib
import time
from argparse import ArgumentParser
from collections import defaultdict

from utils import fastmri
from utils.fastmri import save_reconstructions
import numpy as np
import torch
from utils import fastmri
from utils.fastmri.data import transforms
from utils.fastmri import tensor_to_complex_np
from utils.fastmri.data import SliceDataset
from utils.fastmri.data import transforms as T
from utils.fastmri.data.subsample import MaskFunc
import sigpy.mri as mr
import random
from skimage.metrics import peak_signal_noise_ratio
from utils.fastmri.utils import generate_gro_mask

count_num = 0

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

def cs_total_variation(args, kspace, mask, slice, wt):
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
    kspace = kspace.unsqueeze(0)
    mask = mask.permute(0,2,1)
    kspace = tensor_to_complex_np(kspace)
    mask = mask.cpu().numpy()

    # device = sp.Device(args.device)
    # Estimate sensitivity maps
    # if args.challenge == 'singlecoil':
    #     sens_maps = np.ones(kspace.shape)
    # else:
    sens_maps = mr.app.EspiritCalib(kspace).run()

    # use Total Variation Minimization to reconstruct the image
    pred = mr.app.TotalVariationRecon(kspace, sens_maps, wt, max_iter=args.num_iters).run()
    pred = torch.from_numpy(np.abs(pred))

    return pred


def save_outputs(outputs, output_path):
    """Saves reconstruction outputs to output_path."""
    reconstructions = defaultdict(list)
    times = defaultdict(list)
    
    for fname, slice_num, pred, recon_time in outputs:
        reconstructions[fname].append((slice_num, pred))
        times[fname].append((slice_num, recon_time))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    save_reconstructions(reconstructions, output_path)

    with open('out/recon_times.pkl', 'wb') as f:
        pickle.dump(times, f)


def run_model(idx, reg_wt):
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
    masked_kspace, mask, target, fname, slice_num = dataset[idx]

    prediction = cs_total_variation(
        args, masked_kspace, mask, slice_num, reg_wt
    )

    recon_time = time.perf_counter() - start_time
    return fname, slice_num, prediction, recon_time

def run_model_grid_search(idx, reg_wt):
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
    masked_kspace, mask, target, fname, slice_num = dataset[idx]

    prediction = cs_total_variation(
        args, masked_kspace, mask, slice_num, reg_wt
    )

    recon_time = time.perf_counter() - start_time
    return target, prediction

def run_cs(args):
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        num_slices = len(dataset)
        print(f'RUNNING MODEL ON {num_slices} SLICES')
        for i in range(len(dataset)):
            outputs.append(run_model(i, 0.25*1e-4))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(dataset)))
            time_taken = time.perf_counter() - start_time
            print('DONE - WITH')

    print(f"Run Time = {time_taken} s")
    save_outputs(outputs, args.output_path)

def evaluate_outputs(gt, pred):
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def run_cs_grid_search(args):
    reg_wts = [0.25*1e-4,0.25*1e-5, 0.5*1e-5, 1e-5, 2*1e-5, 4*1e-5]
    avg_psnr = []
    median_psnr = []
    for reg_wt in reg_wts:
        psnr = []
        for i in range(len(dataset)):
            pred, target = run_model_grid_search(i, reg_wt)
            pred = transforms.to_tensor(pred)
            pred = transforms.center_crop(pred, (320,320)).numpy()
            psnr.append(evaluate_outputs(target.numpy(), pred))

        avg_psnr.append(np.mean(psnr))
        median_psnr.append(np.median(psnr))
     
    print(f'Best Reg Wt according to avg PSNR = {reg_wts[np.argmax(avg_psnr)]} - index = {np.argmax(avg_psnr)}')
    print(f'Best Reg Wt according to median PSNR = {reg_wts[np.argmax(median_psnr)]} - index = {np.argmax(median_psnr)}')


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

if __name__ == "__main__":
    parser = create_arg_parser()
    parser.add_argument('--grid-search', default=False)
    parser.add_argument('--output_path', type=pathlib.Path, default=pathlib.Path('out'),
	                help='Path to save the reconstructions to')
    parser.add_argument('--num-iters', type=int, default=200,
	                help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--reg-wt', type=float, default=1e-10,
	                help='Regularization weight parameter')
    parser.add_argument('--num-procs', type=int, default=0,
	                help='Number of processes. Set to 0 to disable multiprocessing.')
    parser.add_argument('--device', type=int, default=0,
	                help='Cuda device idx (-1 for CPU)')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    dataset = SliceDataset(
	root=args.data_path / f'singlecoil_test',
	transform=DataTransform(dev_mask),
	challenge='singlecoil',
	sample_rate=args.sample_rate
    )
    if args.grid_search:
        run_cs_grid_search(args)
    else:
        run_cs(args)
