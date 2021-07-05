#### THIS FILE IS USED FOR TESTING CONCEPTS/IDEAS BEFORE IMPLEMENTATION
import xml.etree.ElementTree as etree
import numpy as np

from utils import fastmri
from utils.fastmri.utils import generate_gro_mask
import h5py
from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import et_query
import torch
import matplotlib.pyplot as plt
import bart
import time

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[3])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

def test_zero_fill():
    with h5py.File('file_brain_AXFLAIR_200_6002441.h5', "r") as hf:
        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        kspace = transforms.to_tensor(hf['kspace'][()])

        # extract target image width, height from ismrmrd header
        enc = ["encoding", "encodedSpace", "matrixSize"]
        crop_size = (
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["y"])),
        )

        print(crop_size)

        # inverse Fourier Transform to get zero filled solution
        mask = get_gro_mask(kspace.shape)
        masked_kspace = kspace * mask + 0.0

        slice_image = fastmri.ifft2c(masked_kspace)
        gt_slice_image = fastmri.ifft2c(kspace)

        # check for FLAIR 203
        if slice_image.shape[-2] < crop_size[1]:
            crop_size = (slice_image.shape[-2], slice_image.shape[-2])

        # crop input image
        image = transforms.complex_center_crop(slice_image, crop_size)
        gt_image = transforms.complex_center_crop(gt_slice_image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)
        gt_image = fastmri.complex_abs(gt_image)

        # apply Root-Sum-of-Squares if multicoil data
        image = fastmri.rss(image, dim=1)
        # image = transforms.center_crop(image, (320, 320))

        gt_image = fastmri.rss(gt_image, dim=1)
        # gt_image = transforms.center_crop()

        #Print first slice of both GT and R=4 images
        fig = plt.figure()
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(np.abs(gt_image.numpy()[0]), cmap='gray')
        plt.xlabel('GT')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(np.abs(image.numpy()[0]), cmap='gray')
        plt.xlabel('R=4 GRO Downsampled')
        plt.show()

def test_cs_tv():
    with h5py.File('file_brain_AXFLAIR_200_6002441.h5', "r") as hf:
        start_time = time.perf_counter()
        #Run CS recon on 1st slice
        et_root = etree.fromstring(hf["ismrmrd_header"][()])
        kspace_gt = transforms.to_tensor(hf['kspace'][()])

        # extract target image width, height from ismrmrd header
        enc = ["encoding", "encodedSpace", "matrixSize"]
        crop_size = (
            int(et_query(et_root, enc + ["y"])),
            int(et_query(et_root, enc + ["y"])),
        )

        # inverse Fourier Transform to get zero filled solution
        mask = get_gro_mask(kspace_gt.shape)
        masked_kspace = kspace_gt * mask + 0.0
        slice_kspace = masked_kspace[0]

        kspace = slice_kspace.permute(1, 2, 0, 3).unsqueeze(0)
        kspace = fastmri.tensor_to_complex_np(kspace)

        sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)
        pred = bart.bart(
            1, f"pics -d0 -S -R T:7:0:0.01 -i 200", kspace, sens_maps
        )
        pred = torch.from_numpy(np.abs(pred[0]))

        # check for FLAIR 203
        if pred.shape[1] < crop_size[1]:
            crop_size = (pred.shape[1], pred.shape[1])

        pred = transforms.center_crop(pred, crop_size)
        total_time = time.perf_counter() - start_time

        print("Recon Time for 200 iterations w/ TV: " + total_time)
        image = fastmri.ifft2c(pred)
        gt_image = fastmri.ifft2c(kspace_gt)

        # absolute value
        image = fastmri.complex_abs(image)
        gt_image = fastmri.complex_abs(gt_image)

        # apply Root-Sum-of-Squares if multicoil data
        image = fastmri.rss(image, dim=1)
        # image = transforms.center_crop(image, (320, 320))

        gt_image = fastmri.rss(gt_image, dim=1)
        # gt_image = transforms.center_crop()

        # Print first slice of both GT and R=4 images
        fig = plt.figure()
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(np.abs(gt_image.numpy()[0]), cmap='gray')
        plt.xlabel('GT')
        fig.add_subplot(rows, columns, 2)
        plt.imshow(np.abs(image.numpy()[0]), cmap='gray')
        plt.xlabel('R=4 GRO Downsampled')
        plt.show()

test_cs_tv()