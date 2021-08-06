import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils import fastmri
import h5py
from utils.fastmri.data import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

rows = 3
cols = 5

def get_psnr(gt, pred):
    maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def get_snr(target, pred):
    noise_mse = np.mean((target - pred)**2)
    return 10*np.log10(np.mean(target**2)/noise_mse)

def get_ssim(target, pred):
    maxval = target.max()
    return structural_similarity(
        target, pred, data_range=maxval
    )

def generate_image(fig, target, image, method, image_ind):
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    if method != 'GT':
        psnr = get_psnr(target, image)
        snr = get_snr(target, image)
        ssim = get_ssim(target, image)
        ax.set_title(f'PSNR: {psnr:.2f}, SNR: {snr:.2f}\nSSIM: {ssim:.4f}')
    ax.imshow(np.abs(image), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method} Reconstruction')

def generate_error_map(fig, target, recon, method, image_ind, relative=False, k=2):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind) # Add to subplot

    # Normalize error between target and reconstruction
    error = (target - recon) if relative else np.abs(target - recon)
    normalized_error = error / error.max()

    im = ax.imshow(k * normalized_error, cmap='bwr' if relative else 'jet', vmin=0, vmax=1) # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Assign x label for plot
    plt.xlabel(f'{method} Relative Error' if relative else f'{method} Absolute Error')

    # Return plotted image and its axis in the subplot
    return im, ax

def get_colorbar(fig, im, ax):
    fig.subplots_adjust(right=0.85) # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.02
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

    fig.colorbar(im, cax=cbar_ax) # Generate colorbar


# h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/pnp/out/{file_name}') as pnp_im, \
data_dir = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
count =0
for fname in tqdm(list(data_dir.glob("*.h5"))):
    count=count+1
    file_name = fname.name
    with h5py.File(fname, "r") as target, \
            h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/base_cs/out/{file_name}', 'r') as recons, \
                h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/zero_filled/out/{file_name}') as zf, \
                    h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/unet/out/{file_name}') as unet_im, \
                        h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/pnp/out/{file_name}') as pnp_im:
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
        pnp_im = pnp_im["reconstruction"][()][ind]
        unet_im = unet_im["reconstruction"][()][ind]

        if need_cropped:
            zfr = transforms.center_crop(zfr, crop_size)
            recons = transforms.center_crop(recons, crop_size)
            pnp_im = transforms.center_crop(pnp_im, crop_size)
            unet_im = transforms.center_crop(unet_im, crop_size)

        gt_max = target.max()
        fig = plt.figure(figsize=(18,9))
        fig.suptitle('T2 Reconstructions')
        generate_image(fig, target, target, 'GT', 1)
        generate_image(fig, target, zfr, 'ZFR', 2)
        generate_image(fig, target, recons, 'CS-TV', 3)
        generate_image(fig, target, unet_im, 'U-Net', 4)
        generate_image(fig, target, pnp_im, 'PnP', 5)

        generate_error_map(fig, target, zfr, 'ZFR', 7)
        generate_error_map(fig, target, recons, 'CS-TV', 8)
        generate_error_map(fig, target, unet_im, 'U-Net', 9)
        im, ax = generate_error_map(fig, target, pnp_im, 'PnP', 10)
        get_colorbar(fig, im, ax)

        generate_error_map(fig, target, zfr, 'ZFR', 12)
        generate_error_map(fig, target, recons, 'CS-TV', 13)
        generate_error_map(fig, target, unet_im, 'U-Net', 14)
        im, ax = generate_error_map(fig, target, pnp_im, 'PnP', 15)
        get_colorbar(fig, im, ax)

        plt.savefig(f'/home/bendel.8/Git_Repos/ComparisonStudy/plots/images/recons_{count}.png')

