"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import xml.etree.ElementTree as etree
import numpy as np
from pathlib import Path

from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import et_query
from tqdm import tqdm
import torch

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

def test_zero_filled(data_dir, out_dir):
    reconstructions = {}

    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            kspace = transforms.to_tensor(hf['kspace'][()])
            # extract target image width, height from ismrmrd header
            crop_size = (320,320)

            # inverse Fourier Transform to get zero filled solution
            masked_kspace = kspace * get_gro_mask(kspace.shape) + 0.0
            slice_image = fastmri.ifft2c(masked_kspace)

            # crop input image
            image = transforms.complex_center_crop(slice_image, crop_size)

            # absolute value
            image = fastmri.complex_abs(image)

            reconstructions[fname.name] = image

    fastmri.save_reconstructions(reconstructions, out_dir)

data = Path('/storage/fastMRI/data/singlecoil_val')
data = Path('/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test/singlecoil_val')
out = Path('out')

test_zero_filled(data, out)
