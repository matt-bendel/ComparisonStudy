import h5py

from utils.fastmri.data.transforms import tensor_to_complex_np
from utils.fastmri.data import transforms
from tqdm import tqdm
from utils import fastmri
from argparse import ArgumentParser
from pathlib import Path
from torch.nn import functional as F

def test(in_path):
    print('BEGINNING PROCESS')
    for fname in tqdm(list(in_path.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            kspace = transforms.to_tensor(hf['kspace'][()])
            image = fastmri.ifft2c(kspace)
            print(image.shape)


data_path = Path('/storage/fastMRI_brain/data/matt_preprocessed_data/singlecoil_val')

test(data_path)
