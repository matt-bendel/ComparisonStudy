import numpy as np

from utils import fastmri
from utils.fastmri.data.transforms import tensor_to_complex_np
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_psnr(gt, pred):
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

# h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/pnp/out/{file_name}') as pnp_im, \
data_dir = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
for fname in tqdm(list(data_dir.glob("*.h5"))):
    file_name = fname.name
    with h5py.File(fname, "r") as target, \
            h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/base_cs/out/{file_name}', 'r') as recons, \
                h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/zero_filled/out/{file_name}') as zf, \
                    h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/unet/out/{file_name}') as unet_im:

        ind = transforms.to_tensor(target['kspace'][()]).shape[0] // 2
        need_cropped = False
        crop_size = (320, 320)
        target = target['reconstruction_rss'][()][ind]
        if target.shape[-1] < 320 or target.shape[-2] < 320:
            need_cropped = True
            crop_size = (target.shape[-1], target.shape[-1]) if target.shape[-1] < target.shape[-2] else (target.shape[-2], target.shape[-2])

        target = transforms.center_crop(target, crop_size).numpy()
        zfr = zf["reconstruction"][()][ind]
        recons = recons["reconstruction"][()][ind]
        # pnp_im = pnp_im["reconstruction"][()][ind]
        unet_im = unet_im["reconstruction"][()][ind]

        if need_cropped:
            zfr = transforms.center_crop(zfr, crop_size)
            recons = transforms.center_crop(recons, crop_size)
            # pnp_im = transforms.center_crop(pnp_im, crop_size)
            unet_im = transforms.center_crop(unet_im, crop_size)


        fig = plt.figure()
        fig.title('T2 Reconstructions')
        ax2 = fig.add_subplot(1, 5, 1)
        ax2.imshow(np.abs(target), cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.xlabel('Ground Truth')

        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(np.abs(zfr), cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.xlabel('ZFR')

        ax3 = fig.add_subplot(1,5,3)
        ax3.title(get_psnr(target, recons))
        ax3.imshow(np.abs(recons), cmap='gray')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.xlabel('CS-TV')

        #ax4 = fig.add_subplot(1, 5, 4)
        #ax4.title(get_psnr(target, pnp_im))
        #ax4.imshow(np.abs(pnp_im), cmap='gray')
        #ax4.set_xticks([])
        #ax4.set_yticks([])
        #plt.xlabel('PnP (RED-GD)')

        ax5 = fig.add_subplot(1, 5, 4)
        ax5.title(get_psnr(target, unet_im))
        ax5.imshow(np.abs(unet_im), cmap='gray')
        ax5.set_xticks([])
        ax5.set_yticks([])
        plt.xlabel('Base Image U-Net')

        plt.savefig(f'/home/bendel.8/Git_Repos/ComparisonStudy/plots/images/{file_name}_recons.png')

