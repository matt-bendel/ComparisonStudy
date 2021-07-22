import numpy as np

from utils import fastmri
import h5py
from utils.fastmri.data import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_psnr(gt, pred):
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def get_snr(target, pred):
    noise = np.abs(target - pred)
    return 20*np.log10(np.mean(target)/np.mean(noise))

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
        k = 7

        fig = plt.figure(figsize=(12,6))
        fig.suptitle('T2 Reconstructions')
        ax2 = fig.add_subplot(2, 4, 1)
        ax2.imshow(np.abs(target), cmap='gray', extent=[0, gt_max, 0, gt_max])
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.xlabel('Ground Truth')

        ax2 = fig.add_subplot(2, 4, 2)
        psnr = get_psnr(target, zfr)
        snr = get_snr(target, zfr)
        ax2.set_title(f'PSNR: {psnr:.2f}\nSNR: {snr:.2f}')
        ax2.imshow(np.abs(zfr), cmap='gray', extent=[0, gt_max, 0, gt_max])
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.xlabel('ZFR')

        ax3 = fig.add_subplot(2,4,3)
        psnr = get_psnr(target, recons)
        snr = get_snr(target, recons)
        ax3.set_title(f'PSNR: {psnr:.2f}\nSNR: {snr:.2f}')
        ax3.imshow(np.abs(recons), cmap='gray', extent=[0, gt_max, 0, gt_max])
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.xlabel('CS-TV')

        #ax4 = fig.add_subplot(1, 5, 4)
        #ax4.title(get_psnr(target, pnp_im))
        #ax4.imshow(np.abs(pnp_im), cmap='gray')
        #ax4.set_xticks([])
        #ax4.set_yticks([])
        #plt.xlabel('PnP (RED-GD)')

        ax5 = fig.add_subplot(2, 4, 4)
        psnr = get_psnr(target, unet_im)
        snr = get_snr(target, unet_im)
        ax5.set_title(f'PSNR: {psnr:.2f}\nSNR: {snr:.2f}')
        ax5.imshow(np.abs(unet_im), cmap='gray', extent=[0, gt_max, 0, gt_max])
        ax5.set_xticks([])
        ax5.set_yticks([])
        plt.xlabel('U-Net')

        ax4 = fig.add_subplot(2, 4, 6)
        ax4.imshow(k*np.abs(target - zfr), cmap='jet', extent=[0, gt_max, 0, gt_max])
        ax4.set_xticks([])
        ax4.set_yticks([])
        plt.xlabel('ZFR Error')

        ax6 = fig.add_subplot(2, 4, 7)
        ax6.imshow(k*np.abs(target-recons), cmap='jet', extent=[0, gt_max, 0, gt_max])
        ax6.set_xticks([])
        ax6.set_yticks([])
        plt.xlabel('CS-TV Error')

        ax7 = fig.add_subplot(2, 4, 8)
        ax7.imshow(k*np.abs(target-unet_im), cmap='jet', extent=[0, gt_max, 0, gt_max])
        ax7.set_xticks([])
        ax7.set_yticks([])
        plt.xlabel('U-Net Error')

        plt.savefig(f'/home/bendel.8/Git_Repos/ComparisonStudy/plots/images/recons_{count}.png')

