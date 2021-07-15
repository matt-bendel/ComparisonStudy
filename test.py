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

def general():
    with h5py.File('/storage/fastMRI_brain/data/Matt_preprocessed_data/singlecoil_val/file_brain_AXFLAIR_200_6002581.h5', "r") as target, \
            h5py.File('/home/bendel.8/Git_Repos/ComparisonStudy/base_cs/out/file_brain_AXFLAIR_200_6002581.h5', 'r') as recons, \
                h5py.File('/home/bendel.8/Git_Repos/ComparisonStudy/zero_filled/out/file_brain_AXFLAIR_200_6002581.h5') as zf:
        ind = 8
        need_cropped = False
        crop_size = (320, 320)
        target = target['reconstruction_rss'][()][ind]
        if target.shape[-1] < 320 or target.shape[-2] < 320:
            need_cropped = True
            crop_size = (target.shape[-1], target.shape[-1]) if target.shape[-1] < target.shape[-2] else (target.shape[-2], target.shape[-2])

        target = transforms.center_crop(target, crop_size)
        zfr = zf["reconstruction"][()][ind]
        recons = recons["reconstruction"][()][ind]
        if need_cropped:
            zfr = transforms.center_crop(zfr, crop_size)
            recons = transforms.center_crop(recons, crop_size)


        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(np.abs(target), cmap='gray')
        plt.xlabel('GT')
        ax1 = fig.add_subplot(1, 3, 2)
        ax1.imshow(np.abs(zfr), cmap='gray')
        plt.xlabel('ZFR')
        ax2 = fig.add_subplot(1,3,3)
        ax2.imshow(np.abs(recons), cmap='gray')
        plt.xlabel('CS-TV')
        # ax3 = fig.add_subplot(1, 3, 3)
        # ax3.imshow(np.abs(usamp_image.numpy()), cmap='gray')
        # plt.xlabel('ZFR')
        plt.savefig('test_graph.png')

general()
