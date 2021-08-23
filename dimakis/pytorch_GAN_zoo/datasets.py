# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import h5py
import pickle

import math
import numpy as np
import imageio

from models.utils.utils import printProgressBar
from models.utils.image_transform import NumpyResize, pil_loader


def saveImage(path, image):
    return imageio.imwrite(path, image)


def prep_mri(inputPath, outputPath):
    in_path = Path(inputPath)
    count = 1
    for fname in tqdm(list(in_path.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            kspace = transforms.to_tensor(hf['kspace'][()])
            image = fastmri.ifft2c(kspace)
            image = fastmri.complex_abs(image).numpy()

            for i in range(image.shape[0]):
                path = os.path.join(outputPath, f'mri_{count}.jpg')
                count = count + 1
                saveImage(path, image[i])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('dataset_name', type=str,
                        choices=['mri'],
                        help='Name of the dataset.')
    parser.add_argument('dataset_path', type=str, default='/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_train',
                        help='Path to the input dataset')
    parser.add_argument('-o', default='/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/dimakis_train', help="If it applies, output dataset (mandadory \
                        for celeba_cropped)",
                        type=str, dest="output_dataset")
    parser.add_argument('-m', dest='model_type',
                        type=str, default='PGAN',
                        choices=['PGAN', 'DCGAN'],
                        help="Model type. Default is progressive growing \
                        (PGAN)")

    args = parser.parse_args()
    prep_mri(args.dataset_path, args.output_dataset)
    config = {"pathDB": args.output_dataset}
    config["config"] = {}


    pathConfig = "config_mri.json"
    with open(pathConfig, 'w') as file:
        json.dump(config, file, indent=2)
