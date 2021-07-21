"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from pathlib import Path

from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
from tqdm import tqdm
import torch

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
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

            # inverse Fourier Transform to get zero filled solution
            masked_kspace = kspace * get_gro_mask(kspace.shape) + 0.0
            image = fastmri.ifft2c(masked_kspace)

            # absolute value
            image = fastmri.complex_abs(image)

            reconstructions[fname.name] = image

    fastmri.save_reconstructions(reconstructions, out_dir)

data = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
# data = Path('/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test_dir/preprocessed/singlecoil_val')
out = Path('out')

test_zero_filled(data, out)
