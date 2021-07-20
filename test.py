#### THIS FILE IS USED FOR TESTING CONCEPTS/IDEAS BEFORE IMPLEMENTATION
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
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

def general(file_type):
    if file_type == 'FLAIR':
        file_name = 'file_brain_AXFLAIR_200_6002581.h5'
    
    if file_type == 'T1':
        file_name = 'file_brain_AXT1_202_6000296.h5'
    
    if file_type == 'T2':
        file_name = 'file_brain_AXT2_200_2000414.h5'

    if file_type == 'T1-WC':
        file_name = 'file_brain_AXT1POST_205_6000151.h5'

    with h5py.File(f'/storage/fastMRI_brain/data/Matt_preprocessed_data/singlecoil_val/{file_name}', "r") as target, \
            h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/base_cs/out/{file_name}', 'r') as recons, \
                h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/zero_filled/out/{file_name}') as zf, \
                    h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/pnp/out/{file_name}') as pnp_im, \
                        h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/unet/out/{file_name}') as unet_im:
        ind = 5
        need_cropped = False
        crop_size = (320, 320)
        target = target['reconstruction_rss'][()][ind]
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


        plt.figure(figsize=(6,6))
        plt.imshow(np.abs(target), cmap='gray')
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom= False)
        plt.xlabel('Ground Truth')
        plt.savefig(f'GT_{file_type}.png')

        fig = plt.figure(figsize=(6,6))
        ax2 = fig.add_subplot(2, 2, 1)
        ax2.imshow(np.abs(zfr), cmap='gray')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.xlabel('ZFR')

        ax3 = fig.add_subplot(2,2,2)
        ax3.imshow(np.abs(recons), cmap='gray')
        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.xlabel('CS-TV')

        ax4 = fig.add_subplot(2, 2, 3)
        ax4.imshow(np.abs(pnp_im), cmap='gray')
        ax4.set_xticks([])
        ax4.set_yticks([])
        plt.xlabel('PnP (RED-GD)')

        ax5 = fig.add_subplot(2, 2, 4)
        ax5.imshow(np.abs(unet_im), cmap='gray')
        ax5.set_xticks([])
        ax5.set_yticks([])
        plt.xlabel('Base Image U-Net')

        plt.savefig(f'{file_type}_RECONS.png')

general('T1-WC')
