import argparse
import pathlib
from argparse import ArgumentParser
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import fastmri
from utils.fastmri.data import transforms
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pickle

all_psnr = []
all_ssim = []
all_snr = []
all_ratio = []

def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    psnr_val = peak_signal_noise_ratio(gt, pred, data_range=maxval)
    all_psnr.append(psnr_val)
    return psnr_val

def snr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute the Signal to Noise Ratio metric (SNR)"""
    noise_mse = np.mean((gt - pred)**2)
    snr = 10*np.log10(np.mean(gt**2)/noise_mse)
    all_snr.append(snr)
    return snr

def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    #if not gt.ndim == 3:
     #   raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = structural_similarity(
            gt, pred, data_range=maxval
        )
    #for slice_num in range(gt.shape[0]):
     #   ssim = ssim + structural_similarity(
      #      gt[slice_num], pred[slice_num], data_range=maxval
       # )
    #val = ssim / gt.shape[0]
    all_ssim.append(ssim)

    return ssim


METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    SNR=snr,
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {metric: stat.mean() for metric, stat in self.metrics.items()}

    def stddevs(self):
        return {metric: stat.stddev() for metric, stat in self.metrics.items()}

    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        medians = {'MSE': 0, 'NMSE': 0, 'PSNR': np.median(all_psnr), 'SSIM': np.median(all_ssim), 'SNR': np.median(all_snr)}
        metric_names = sorted(list(means))
        return " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}, Median = {medians[name]}\n"
            for name in metric_names
        )

def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    for tgt_file in args.target_path.iterdir():
        with h5py.File(tgt_file, "r") as target, h5py.File(
            args.predictions_path / tgt_file.name, "r"
        ) as recons:
            if args.acquisition and args.acquisition != target.attrs["acquisition"]:
                continue
            target = transforms.to_tensor(target['kspace'][()])
            target = fastmri.ifft2c(target)
            target = fastmri.complex_abs(target).numpy()
            recons = np.squeeze(np.squeeze(recons["reconstruction"][()],axis=-1),axis=1) * np.max(target) / 2#recons["reconstruction"][()]
            for i in range(target.shape[0]):
                metrics.push(target[i], recons[i])
                all_ratio.append(recons / np.linalg.norm(target[i],2))

    return metrics

def get_avg_slice_time(args):
    with open(args.predictions_path / 'recon_times.pkl', 'rb') as f:
        times = pickle.load(f)
        total_time = 0
        num_volumes = len(times)
        num_slices = 0
        for file_name in times:
            volume = times[file_name]
            num_slices = num_slices + len(volume)
            if args.method == 'cs':
                total_time = np.sum(volume, axis=0)[1] + total_time
            else:
                total_time = np.sum(volume) + total_time
        
        print(f'Average time per slice: {total_time/num_slices}')
        print(f'Average time per volume: {total_time/num_volumes}')
            
def save_histogram(metric_name, metric_list, method):
    x = np.array(metric_list)
    plt.hist(x, density=True,bins=30)
    if metric_name == 'PSNR':
        plt.xlim(20, 40)
    elif metric_name == 'SNR':
        plt.xlim(10, 30)
    else:
        plt.xlim(0.7,1)
    plt.title(f'Histogram of {metric_name} from {method} reconstructioon')
    plt.xlabel(metric_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'/home/bendel.8/Git_Repos/ComparisonStudy/plots/graphs/{method}_{metric_name}.png')

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--target-path",
        type=pathlib.Path,
        required=True,
        help="Path to the ground truth data",
    )
    parser.add_argument(
        "--predictions-path",
        type=pathlib.Path,
        required=True,
        help="Path to reconstructions",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method being evaluated",
    )
    parser.add_argument(
        "--challenge",
        choices=["singlecoil", "multicoil"],
        help="Which challenge",
        default='singlecoil'
    )
    parser.add_argument("--acceleration", type=int, default=None)
    parser.add_argument(
        "--acquisition",
        choices=[
            "CORPD_FBK",
            "CORPDFS_FBK",
            "AXT1",
            "AXT1PRE",
            "AXT1POST",
            "AXT2",
            "AXFLAIR",
        ],
        default=None,
        help="If set, only volumes of the specified acquisition type are used "
        "for evaluation. By default, all volumes are included.",
    )
    args = parser.parse_args()

    metrics = evaluate(args, 'reconstruction_rss')
    print(metrics)
    get_avg_slice_time(args)
    #save_histogram('PSNR', all_psnr, args.method)
    #save_histogram('SNR', all_snr, args.method)
    save_histogram('Ratio', all_ratio, args.method)
    #save_histogram('SSIM', all_ssim, args.method)
