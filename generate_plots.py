import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from utils import fastmri
import h5py
from utils.fastmri.data import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

rows = 2
cols = 4

def get_psnr(gt, pred):
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def get_snr(target, pred):
    noise = np.abs(target - pred)
    return 20*np.log10(np.mean(target)/np.mean(noise))

def generate_image(fig, max, target, image, method, image_ind):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT':
        psnr = get_psnr(target, image)
        snr = get_snr(target, image)
        ax.set_title(f'PSNR: {psnr:.2f}\nSNR: {snr:.2f}')
    ax.imshow(np.abs(image), cmap='gray', extent=[0, max, 0, max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method} Reconstruction')

def generate_error_map(fig, max, target, recon, method, image_ind, k=5):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    ax.imshow(k * np.abs(target - recon), cmap='jet', extent=[0, max, 0, max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method} Error')

# h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/pnp/out/{file_name}') as pnp_im, \
data_dir = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
count =0
for fname in tqdm(list(data_dir.glob("*.h5"))):
    count=count+1
    file_name = fname.name
    with h5py.File(fname, "r") as target, \
            h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/base_cs/out/{file_name}', 'r') as recons, \
                h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/zero_filled/out/{file_name}') as zf, \
                    h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/unet/out/{file_name}') as unet_im:
        ind = transforms.to_tensor(target['kspace'][()]).shape[0] // 2
        need_cropped = False
        crop_size = (320, 320)
        target = transforms.to_tensor(target['kspace'][()])
        target = fastmri.ifft2c(target)
        target = fastmri.complex_abs(target).numpy()[ind]
        if target.shape[-1] < 320 or target.shape[-2] < 320:
            need_cropped = True
            crop_size = (target.shape[-1], target.shape[-1]) if target.shape[-1] < target.shape[-2] else (target.shape[-2], target.shape[-2])

        target = transforms.center_crop(target, crop_size)
        zfr = zf["reconstruction"][()][ind]
        recons = recons["reconstruction"][()][ind]
        # pnp_im = pnp_im["reconstruction"][()][ind]
        unet_im = unet_im["reconstruction"][()][ind]

        if need_cropped:
            zfr = transforms.center_crop(zfr, crop_size)
            recons = transforms.center_crop(recons, crop_size)
            # pnp_im = transforms.center_crop(pnp_im, crop_size)
            unet_im = transforms.center_crop(unet_im, crop_size)

        gt_max = target.max()
        fig = plt.figure(figsize=(12,6))
        fig.suptitle('T2 Reconstructions')
        generate_image(fig, gt_max, target, target, 'GT', 1)
        generate_image(fig, gt_max, target, zfr, 'ZFR', 2)
        generate_image(fig, gt_max, target, recons, 'CS-TV', 3)
        generate_image(fig, gt_max, target, unet_im, 'U-Net', 4)
        generate_error_map(fig, gt_max, target, zfr, 'ZFR', 6)
        generate_error_map(fig, gt_max, target, recons, 'CS-TV', 7)
        generate_error_map(fig, gt_max, target, unet_im, 'U-Net', 8)

        plt.savefig(f'/home/bendel.8/Git_Repos/ComparisonStudy/plots/images/recons_{count}.png')
