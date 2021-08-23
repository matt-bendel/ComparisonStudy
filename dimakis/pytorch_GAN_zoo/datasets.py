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


def celebaSetup(inputPath,
                outputPath,
                pathConfig="config_celeba_cropped.json"):

    imgList = [f for f in os.listdir(
        inputPath) if os.path.splitext(f)[1] == ".jpg"]
    cx = 89
    cy = 121

    nImgs = len(imgList)

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    for index, item in enumerate(imgList):
        printProgressBar(index, nImgs)
        path = os.path.join(inputPath, item)
        img = np.array(pil_loader(path))

        img = img[cy - 64: cy + 64, cx - 64: cx + 64]

        path = os.path.join(outputPath, item)
        saveImage(path, img)

    printProgressBar(nImgs, nImgs)


def resizeDataset(inputPath, outputPath, maxSize):

    sizes = [64, 128, 512, 1024]
    scales = [0, 5, 6, 8]
    index = 0

    imgList = [f for f in os.listdir(inputPath) if os.path.splitext(f)[
        1] in [".jpg", ".npy"]]

    nImgs = len(imgList)

    if maxSize < sizes[0]:
        raise AttributeError("Maximum resolution too low")

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    datasetProfile = {}

    for index, size in enumerate(sizes):

        if size > maxSize:
            break

        localPath = os.path.join(outputPath, str(size))
        if not os.path.isdir(localPath):
            os.mkdir(localPath)

        datasetProfile[str(scales[index])] = localPath

        print("Resolution %d x %d" % (size, size))

        resizeModule = NumpyResize((size, size))

        for index, item in enumerate(imgList):
            printProgressBar(index, nImgs)
            path = os.path.join(inputPath, item)
            img = pil_loader(path)

            img = resizeModule(img)
            path = os.path.splitext(os.path.join(localPath, item))[0] + ".jpg"
            saveImage(path, img)
        printProgressBar(nImgs, nImgs)

    return datasetProfile, localPath


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
