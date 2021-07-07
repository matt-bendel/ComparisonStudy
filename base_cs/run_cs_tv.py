import sigpy.mri as mr

from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    inds = mask_shape[-2]
    mask = generate_gro_mask(1)[0:inds]
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

with h5py.File('/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/file1000001.h5', "r") as hf:
    crop_size = (320, 320)
    kspace = transforms.to_tensor(hf['kspace'][()])[29]
    image = fastmri.ifft2c(kspace)
    image = transforms.complex_center_crop(image, crop_size)
    image = fastmri.complex_abs(image)

    mask = get_gro_mask(kspace.shape)
    usamp_kspace = (kspace * mask + 0.0)
    usamp_image = fastmri.ifft2c(usamp_kspace)
    usamp_image = transforms.complex_center_crop(usamp_image, crop_size)
    usamp_image = fastmri.complex_abs(usamp_image)
    kspace = (kspace * mask + 0.0).unsqueeze(0)

    kspace = fastmri.tensor_to_complex_np(kspace)
    mps = np.ones(kspace.shape)
    lamda=1e-9
    pred = mr.app.TotalVariationRecon(kspace, mps, lamda, max_iter=400).run()
    pred = torch.from_numpy(np.abs(pred))
    pred = transforms.center_crop(pred, crop_size)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(np.abs(image.numpy()), cmap='gray')
    plt.xlabel('GT')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.abs(usamp_image.numpy()), cmap='gray')
    plt.xlabel('ZFR')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(np.abs(pred.numpy()), cmap='gray')
    plt.xlabel('TV Reconstruction')

    plt.show()