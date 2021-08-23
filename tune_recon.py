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

data_dir = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
count = 1
for fname in tqdm(list(data_dir.glob("*.h5"))):
    file_name = fname.name
    rows = 1
    cols = 5
    with h5py.File(fname, "r") as target, \
            h5py.File(f'/home/bendel.8/Git_Repos/ComparisonStudy/cs-mri-gan-master/out/{file_name}', 'r') as recons:
        alpha = np.linspace(0,1,5)
        target = transforms.to_tensor(target['kspace'][()])
        target = fastmri.ifft2c(target)
        target = fastmri.complex_abs(target).numpy()[4]
        gan_im = np.squeeze(np.squeeze(recons["reconstruction"][()][4], axis=0), axis=-1) * np.max(target) / 2

        fig = plt.figure(figsize=(18,9))
        ind = 1
        for num in alpha:
            ax = fig.add_subplot(rows, cols, ind)
            ind = ind + 1
            ax.imshow(np.abs(gan_im), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.xlabel(f'alpha = {num}')

