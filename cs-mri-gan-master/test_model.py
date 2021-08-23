import time

import tensorflow as tf
import os

import torch
from keras.utils import multi_gpu_model 
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation
from keras.layers import LeakyReLU, BatchNormalization, Lambda
import numpy as np
from metrics import metrics
import pickle
from collections import defaultdict
from utils.fastmri import save_reconstructions
from utils.fastmri.data import SliceDataset
from utils.fastmri.data.subsample import MaskFunc
from utils.fastmri.utils import generate_gro_mask
import pathlib
from utils.fastmri.data import transforms as T
from utils import fastmri
import matplotlib.pyplot as plt
from utils.fastmri import tensor_to_complex_np

def resden(x, fil, gr, beta, gamma_init, trainable):
    x1 = Conv2D(filters=gr, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    x1 = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Concatenate(axis=-1)([x, x1])

    x2 = Conv2D(filters=gr, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                bias_initializer='zeros')(x1)
    x2 = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)

    x2 = Concatenate(axis=-1)([x1, x2])

    x3 = Conv2D(filters=gr, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                bias_initializer='zeros')(x2)
    x3 = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x3)
    x3 = LeakyReLU(alpha=0.2)(x3)

    x3 = Concatenate(axis=-1)([x2, x3])

    x4 = Conv2D(filters=gr, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                bias_initializer='zeros')(x3)
    x4 = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(x4)
    x4 = LeakyReLU(alpha=0.2)(x4)

    x4 = Concatenate(axis=-1)([x3, x4])

    x5 = Conv2D(filters=fil, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                bias_initializer='zeros')(x4)
    x5 = Lambda(lambda x: x * beta)(x5)
    xout = Add()([x5, x])

    return xout


def resresden(x, fil, gr, betad, betar, gamma_init, trainable):
    x1 = resden(x, fil, gr, betad, gamma_init, trainable)
    x2 = resden(x1, fil, gr, betad, gamma_init, trainable)
    x3 = resden(x2, fil, gr, betad, gamma_init, trainable)
    x3 = Lambda(lambda x: x * betar)(x3)
    xout = Add()([x3, x])

    return xout


def generator(inp_shape, trainable=False):
    gamma_init = tf.random_normal_initializer(1., 0.02)

    fd = 512
    gr = 32
    nb = 12
    betad = 0.2
    betar = 0.2

    inp_real_imag = Input(inp_shape)
    lay_128dn = Conv2D(64, (4, 4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer='he_normal',
                       bias_initializer='zeros')(inp_real_imag)

    lay_128dn = LeakyReLU(alpha=0.2)(lay_128dn)

    lay_64dn = Conv2D(128, (4, 4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer='he_normal',
                      bias_initializer='zeros')(lay_128dn)
    lay_64dn = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_64dn)
    lay_64dn = LeakyReLU(alpha=0.2)(lay_64dn)

    lay_32dn = Conv2D(256, (4, 4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer='he_normal',
                      bias_initializer='zeros')(lay_64dn)
    lay_32dn = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_32dn)
    lay_32dn = LeakyReLU(alpha=0.2)(lay_32dn)

    lay_16dn = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer='he_normal',
                      bias_initializer='zeros')(lay_32dn)
    lay_16dn = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_16dn)
    lay_16dn = LeakyReLU(alpha=0.2)(lay_16dn)  # 16x16

    lay_8dn = Conv2D(512, (4, 4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer='he_normal',
                     bias_initializer='zeros')(lay_16dn)
    lay_8dn = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_8dn)
    lay_8dn = LeakyReLU(alpha=0.2)(lay_8dn)  # 8x8

    xc1 = Conv2D(filters=fd, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                 bias_initializer='zeros')(lay_8dn)  # 8x8
    xrrd = xc1
    for m in range(nb):
        xrrd = resresden(xrrd, fd, gr, betad, betar, gamma_init, trainable)

    xc2 = Conv2D(filters=fd, kernel_size=3, strides=1, padding='same', use_bias=True, kernel_initializer='he_normal',
                 bias_initializer='zeros')(xrrd)
    lay_8upc = Add()([xc1, xc2])

    lay_16up = Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                               kernel_initializer='he_normal', bias_initializer='zeros')(lay_8upc)
    lay_16up = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_16up)
    lay_16up = Activation('relu')(lay_16up)  # 16x16

    lay_16upc = Concatenate(axis=-1)([lay_16up, lay_16dn])

    lay_32up = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                               kernel_initializer='he_normal', bias_initializer='zeros')(lay_16upc)
    lay_32up = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_32up)
    lay_32up = Activation('relu')(lay_32up)  # 32x32

    lay_32upc = Concatenate(axis=-1)([lay_32up, lay_32dn])

    lay_64up = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                               kernel_initializer='he_normal', bias_initializer='zeros')(lay_32upc)
    lay_64up = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_64up)
    lay_64up = Activation('relu')(lay_64up)  # 64x64

    lay_64upc = Concatenate(axis=-1)([lay_64up, lay_64dn])

    lay_128up = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                                kernel_initializer='he_normal', bias_initializer='zeros')(lay_64upc)
    lay_128up = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_128up)
    lay_128up = Activation('relu')(lay_128up)  # 128x128

    lay_128upc = Concatenate(axis=-1)([lay_128up, lay_128dn])

    lay_256up = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=True,
                                kernel_initializer='he_normal', bias_initializer='zeros')(lay_128upc)
    lay_256up = BatchNormalization(gamma_initializer=gamma_init, trainable=trainable)(lay_256up)
    lay_256up = Activation('relu')(lay_256up)  # 256x256

    out = Conv2D(1, (1, 1), strides=(1, 1), activation='tanh', padding='same', use_bias=True,
                 kernel_initializer='he_normal', bias_initializer='zeros')(lay_256up)

    model = Model(inputs=inp_real_imag, outputs=out)

    return model

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

class DataTransform(object):
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, mask_func):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.mask_func = mask_func

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        kspace = T.to_tensor(kspace)
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask) + 0.0

        target = fastmri.ifft2c(kspace)
        image = fastmri.ifft2c(masked_kspace)

        target = fastmri.complex_abs(target).numpy() 
        
        max_val = np.max(target)

        target = target / max_val * 2
        image = tensor_to_complex_np(image) / max_val * 2

        image = np.expand_dims(image, axis=-1)
        usam_real = image.real
        usam_imag = image.imag

        u_sampled_data_2c = np.concatenate((usam_real, usam_imag), axis=-1)
        return (u_sampled_data_2c, mask, target, fname, slice)

def save_outputs(outputs, output_path):
    """Saves reconstruction outputs to output_path."""
    reconstructions = defaultdict(list)
    times = defaultdict(list)

    for fname, slice_num, pred, recon_time in outputs:
        reconstructions[fname].append((slice_num, pred))
        times[fname].append((slice_num, recon_time))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    save_reconstructions(reconstructions, output_path)

    with open('out/recon_times.pkl', 'wb') as f:
        pickle.dump(times, f)

def test_method(idx, gen):
    zfr, mask, target, fname, slice_num = dataset[idx] 

    start_time = time.perf_counter()
    prediction = gen4.predict(np.expand_dims(zfr, axis=0))

    recon_time = time.perf_counter() - start_time
    psnr, ssim = metrics(np.expand_dims(target,axis=0),prediction[:,:,:,0],2.0)
    print(f'PSNR: {psnr}, SSIM: {ssim}')
    
    return fname, slice_num, prediction, recon_time


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

if __name__ == '__main__':
    data_path = '/storage/fastMRI_brain/data/Matt_preprocessed_data/T2'
    dataset = SliceDataset(
        root=pathlib.Path(data_path) / f'singlecoil_test',
        transform=DataTransform(None),
        challenge='singlecoil',
        sample_rate=1.0
    )

    outputs = []

    inp_shape = (320, 320, 2)
    gen4 = generator(inp_shape=inp_shape, trainable=False)

    filename = '/home/bendel.8/Git_Repos/ComparisonStudy/cs-mri-gan-master/gen_weights_a5_0303.h5'
    gen4.load_weights(filename)

    for i in range(len(dataset)):
        outputs.append(test_method(i, gen4))

    save_outputs(outputs, pathlib.Path('out'))

'''
data_path = 'testing_gt.pickle'  # Ground truth
usam_path = 'testing_usamp.pickle'  # Zero-filled reconstructions

df = open(data_path, 'rb')
uf = open(usam_path, 'rb')

dataset_real = np.asarray(pickle.load(df), dtype='float32')
print(f'REAL IMAGES: {len(dataset_real)}, SHAPE: {dataset_real[0].shape}')

u_sampled_data = np.expand_dims(pickle.load(uf), axis=-1)
usam_real = u_sampled_data.real
usam_imag = u_sampled_data.imag

u_sampled_data_2c = np.concatenate((usam_real, usam_imag), axis=-1)
print(f'USAMP IMAGES: {len(u_sampled_data_2c)}, SHAPE: {u_sampled_data_2c[0].shape}')

inp_shape = (320, 320, 2)
gen4 = generator(inp_shape=inp_shape, trainable=False)

filename = '/home/bendel.8/Git_Repos/ComparisonStudy/cs-mri-gan-master/gen_weights_a5_0303.h5'
gen4.load_weights(filename)

preds = gen4.predict(u_sampled_data_2c)
psnr, ssim = metrics(dataset_real, preds[:,:,:,0], 2.0)
print(psnr)
'''
