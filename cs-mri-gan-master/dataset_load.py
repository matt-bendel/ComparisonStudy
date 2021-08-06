import os
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
import torch
from utils.fastmri import tensor_to_complex_np

save_path=''

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

def load_a(path, num):
        data = []
        usamp_data = []
        for fname in tqdm(list(path.glob("*.h5"))):
            with h5py.File(fname, "r") as hf:
                kspace = transforms.to_tensor(hf['kspace'][()])
                mask = get_gro_mask(kspace.shape)
                usamp_kspace = (kspace * mask) + 0.0

                image = fastmri.ifft2c(kspace)
                usamp_image = fastmri.ifft2c(usamp_kspace)

                image = fastmri.complex_abs(image)

                for i in range(image.shape[0]):
                    slice_gt = image[i].numpy() / np.max(image[i].numpy()) * 2
                    data.append(slice_gt)

                    slice_us = tensor_to_complex_np(usamp_image[i]) / tensor_to_complex_np(usamp_image[i]) * 2
                    usamp_data.append(slice_us)

        return np.asarray(data), np.asarray(usamp_data)

data = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_train')
train_gt, train_us = load_a(data,0)

#data = Path('/storage/fastMRI_brain/data/multicoil_val')
#train_gt_2, train_us_2 = load_a(data,0)

print(f"TOTAL NUMBER OF TRAINING IMAGES: {train_gt.shape[0]}")

with open(os.path.join(save_path,'training_gt.pickle'),'wb') as f:
    pickle.dump(train_gt,f,protocol=4)

with open(os.path.join(save_path,'training_usamp.pickle'),'wb') as f:
    pickle.dump(train_us,f,protocol=4)

'''
#for testing data

test_path = Path('/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test')
test_gt, test_us = load_a(test_path, 0)

with open(os.path.join(save_path,'testing_gt.pickle'),'wb') as f:
       pickle.dump(test_gt, f, protocol=4)

with open(os.path.join(save_path, 'testing_usamp.pickle'), 'wb') as f:
    pickle.dump(test_us, f, protocol=4)
'''
