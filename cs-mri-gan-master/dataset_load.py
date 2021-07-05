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

save_path=''

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[3])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

def load_a(path, num):
        data = []
        usamp_data = []

        for fname in tqdm(list(path.glob("*.h5"))):
            with h5py.File(fname, "r") as hf:
                kspace = transforms.to_tensor(hf['kspace'][()])
                if kspace.shape[3] == 320:
                    mask = get_gro_mask(kspace.shape)
                    usamp_kspace = kspace * mask + 0.0

                    crop_size = (320, 320)

                    slice_image = fastmri.ifft2c(kspace)
                    usamp_slice_image = fastmri.ifft2c(usamp_kspace)

                    # check for FLAIR 203
                    if slice_image.shape[-2] < crop_size[1]:
                        crop_size = (slice_image.shape[-2], slice_image.shape[-2])

                    # crop input image
                    image = transforms.complex_center_crop(slice_image, crop_size)
                    image = fastmri.complex_abs(image)

                    usamp_image = transforms.complex_center_crop(usamp_slice_image, crop_size)

                    # apply Root-Sum-of-Squares if multicoil data
                    image = fastmri.rss(image, dim=1)
                    usamp_image = fastmri.rss(usamp_image, dim=1)

                    for i in range(image.shape[0]):
                        data.append(image[i,:,:].numpy())
                        usamp_data.append(usamp_image[i,:,:,:].numpy())

        return np.asarray(data), np.asarray(usamp_data)

data = Path('/storage/fastMRI_brain/data/multicoil_train')
train_gt, train_us = load_a(data,0)

with open(os.path.join(save_path,'training_gt.pickle'),'wb') as f:
    pickle.dump(train_gt,f,protocol=4)

with open(os.path.join(save_path,'training_usamp.pickle'),'wb') as f:
    pickle.dump(train_us,f,protocol=4)


'''
#for testing data

#miccai dataset
test_path='/home/cs-mri-gan/training-testing/warped-images'
test_data=load_a(test_path, 390)

#mrnet dataset
#test_path='/home/cs-mri-gan/valid/coronal'
#test_data=load_b(test_path)

with open(os.path.join(save_path,'testing_gt.pickle'),'wb') as f:
       pickle.dump(test_data,f,protocol=4)
'''
