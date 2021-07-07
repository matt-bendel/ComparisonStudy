#### THIS FILE IS USED FOR TESTING CONCEPTS/IDEAS BEFORE IMPLEMENTATION
import numpy as np

from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    inds = mask_shape[-2]
    mask = generate_gro_mask(mask_shape[3])[0:inds]
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

def general():
    with h5py.File('file1000001.h5', "r") as hf:
        print('Keys:', list(hf.keys()))
        print('Attrs:', dict(hf.attrs))
        print(hf.attrs['acquisition'][0:6])
        if hf.attrs['acquisition'][0:6] == 'CORPDF':
            print('good')
        exit()
        crop_size = (320,320)
        kspace = transforms.to_tensor(hf['kspace'][()])
        mask = get_gro_mask(kspace.shape)
        usamp_kspace = kspace * mask + 0.0

        image = fastmri.ifft2c(kspace)
        usamp_image = fastmri.ifft2c(usamp_kspace)

        # crop input image
        image = transforms.complex_center_crop(image, crop_size)
        usamp_image = transforms.complex_center_crop(usamp_image, crop_size)

        image = fastmri.complex_abs(image)[30]
        usamp_image = fastmri.complex_abs(usamp_image)[30]

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(np.abs(image.numpy()), cmap='gray')
        plt.xlabel('Calculated GT')
        cropped_image = transforms.to_tensor(hf['reconstruction_esc'][()])[30]
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(np.abs(cropped_image.numpy()),cmap='gray')
        plt.xlabel('Given GT')
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(np.abs(usamp_image.numpy()), cmap='gray')
        plt.xlabel('ZFR')

        plt.show()

general()