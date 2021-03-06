"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
# import multiprocessing
import pathlib
import random
import time
from collections import defaultdict

import numpy as np
import torch
from utils.fastmri.utils import generate_gro_mask

import matplotlib.pyplot as plt

# import bart
import sigpy as sp
import sigpy.mri as mr

from utils.fastmri import utils
from argparse import ArgumentParser
from utils.fastmri import tensor_to_complex_np
from eval import nmse, psnr

from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import SliceDataset

from utils.fastmri.models.PnP.train_denoiser_multicoil_brain import load_model
from utils import fastmri

import scipy.misc
import PIL
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def psnr_2(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    plt.imshow(np.abs(pred),cmap='gray')
    plt.savefig('pred.png')
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

def optimal_scale(target, recons, return_alpha=False):
    if recons.ndim == 3:
        alpha = np.sum(target*recons, axis=(1,2), keepdims=True)/np.sum(recons**2, axis=(1,2), keepdims=True)
    else:
        alpha = np.sum(target*recons, axis=(0,1), keepdims=True) / np.sum(recons**2, axis=(0,1), keepdims=True)
    # print(alpha)
    if return_alpha:
        return alpha * recons, alpha
    return alpha*recons

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

class DataTransform:
    """
    Data Transformer that masks input k-space.
    """

    def __init__(self, args, use_seed=None):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
        """
        self.use_seed = use_seed
        self.args = args
        self.mask = None
        if args.mask_path is not None:
            self.mask = torch.load(args.mask_path)

    def __call__(self, kspace, mk, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array, optional): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                masked_kspace (torch.Tensor): Sub-sampled k-space with the same shape as kspace.
                fname (str): File name containing the current data item
                slice (int): The index of the current slice in the volume
        """
        kspace = transforms.to_tensor(kspace)
        #sens = mr.app.EspiritCalib(tensor_to_complex_np(kspace)).run()
        sens = np.ones((320, 320, 2))
        mask = get_gro_mask(kspace.shape)
        masked_kspace = (kspace * mask) + 0.0

        target = fastmri.ifft2c(kspace).float()
        #image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        #image = image.clamp(-6, 6)
        #
        #target = transforms.normalize(target, mean, std, eps=1e-11)
        #target = target.clamp(-6, 6)

        return masked_kspace, mask, sens, target, fname, slice

def create_data_loader(args):
    # select subset
    data_set = SliceDataset(
        sample_rate=1,
        root=args.data_path / f'singlecoil_test',
        transform=DataTransform(args),
        challenge='singlecoil',
    )
    return data_set

def denoiser(noisy,model,args):
    # add rotate
    noisy, rot_angle = transforms.best_rotate(noisy, args.rotation_angles)

    # normalize
    if (args.normalize == 'max') or (args.normalize == 'std'):
        noisy, scale = transforms.denoiser_normalize(noisy, is_complex=True, use_std=args.normalize=='std')
    elif args.normalize=='constant':
        scale = 0.0016
        noisy = noisy/scale
    else:
        scale = 1

    if args.denoiser_mode=='mag':
        mag = transforms.complex_abs(noisy)
        phase = transforms.phase(noisy)
        denoised_mag = model(mag[None, None,...])
        denoised_mag = denoised_mag[0,0,...]
        denoised_image = transforms.polar_to_rect(denoised_mag, phase)

    elif args.denoiser_mode == '2-chan':
        # move real/imag to channel position
        #noisy, mean, std = transforms.normalize_instance(noisy, eps=1e-11)
        #noisy = noisy.clamp(-6, 6)
        noisy = noisy.float().permute(2,0,1).unsqueeze(0)
        denoised_image = model(noisy).squeeze(0).permute(1,2,0)

    elif args.denoiser_mode == 'real-imag':
        # move real/imag to batch position
        noisy = noisy.permute(2,0,1).unsqueeze(1)
        denoised_image = model(noisy).squeeze(1).permute(1,2,0)

    # unnormalize
    denoised_image = denoised_image*scale

    # undo rotate
    denoised_image = transforms.polar_to_rect(transforms.complex_abs(denoised_image), transforms.phase(denoised_image)-rot_angle)

    return denoised_image

def find_spec_rad(mri,steps, x):
    # init x
    x = torch.randn_like(x)
    x = x/torch.sqrt(torch.sum(torch.abs(x)**2))

    # power iteration
    for i in range(steps):
        x = fastmri.ifft2c(fastmri.fft2c(x))
        spec_rad = torch.sqrt(torch.sum(torch.abs(x)**2))
        x = x/spec_rad

    return spec_rad


def pds_normal(y, model, args, mri, target, max_iter, gamma_1_input, sens_map_foo):
    #     print('Running generic-2 PnP-PDS')

    device = torch.device("cuda")

    EYE = torch.ones(320, 320, 2)
    EYE = EYE.to(device)

    sens_map_zero_count = 0
    AA = torch.zeros(320, 320)
    AA_ones = torch.zeros(320, 320) + 1
    for i in range(320):
        for j in range(320):
            if np.sum(np.abs(sens_map_foo[i, j])) == 0:
                sens_map_zero_count = sens_map_zero_count + 1
                AA[i, j] = 1
                AA_ones[i, j] = 0
    AA = AA.to(device)
    AA_ones = AA_ones.to(device)

    roo = 6.3688e-12

    with torch.no_grad():
        outer_iters = max_iter

        a_nmse = []
        a_rSNR = []
        gamma_1_log = []
        gamma_2_log = []
        res_norm_log = []
        required_res_norm_log = []
        input_RMSE = []
        output_RMSE = []

        x = fastmri.ifft2c(y)
        z = 0 * x

        x_crop = transforms.complex_abs(x)

        L = find_spec_rad(mri, 100, x)

        gamma_1 = gamma_1_input * roo

        gamma_2 = (1 / L) * ((1 / gamma_1))

        for k in range(outer_iters):
            yoda = 1
            b1 = x - (gamma_1 * fastmri.ifft2c(z))

            x_new = yoda * denoiser(b1, model, args) + (1 - yoda) * x

            x_hat = x_new + 1 * (x_new - x)

            z = (EYE - (1 / ((((1 / roo) * EYE) / gamma_2) + EYE))) * z + (
                        1 / ((((1 / roo) * EYE) / gamma_2) + EYE)) * ((1 / roo) * EYE) * (fastmri.fft2c(x_hat) - y)

            x = x_new

            gamma_1_log.append(gamma_1)
            gamma_2_log.append(gamma_2)

            boo = fastmri.fft2c(x) - y

            resnorm_recov = torch.sqrt(torch.sum(boo ** 2))

            x_crop = transforms.complex_abs(x)

            nmse_step = psnr_2(transforms.complex_abs(target), x_crop)
            print(nmse_step)
            a_nmse.append(nmse_step)

            res_norm_log.append(resnorm_recov)

    return x, a_nmse, gamma_1_log, gamma_2_log, required_res_norm_log, res_norm_log, input_RMSE, output_RMSE

class A_mri:
    def __init__(self,sens_maps,mask):
        self.sens_maps = sens_maps
        self.mask = mask
    
    def A(self,x):
        x = x[None, ...]
        y = transforms.complex_mult(x,self.sens_maps)
        y_fft = fastmri.fft2c(y)
        out = self.mask * y_fft
        return out

    def H(self,x):
        y = self.mask*x
        y_ifft = fastmri.ifft2c(y)
        out = torch.sum(transforms.complex_mult(y_ifft,fastmri.complex_conj(self.sens_maps)), dim=0)
        return out
        

def cs_pnp(args, model, kspace, mask, sens, target):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """
    # mask = mask.permute(0,2,1)
    masked_kspace = tensor_to_complex_np(kspace)

    sens_maps = transforms.to_tensor(sens)

    # handle pytorch device
    device = torch.device("cuda")
    sens_maps = None
    mask = mask.to(device)
    masked_kspace = transforms.to_tensor(masked_kspace)
    masked_kspace = masked_kspace.to(device)
    sens_map_foo = np.zeros((320,320)).astype(np.complex)
    mri = A_mri(sens_maps, mask)
    target = target.to(device)

    pred, a_nmse = admm(masked_kspace,model,args,target)
    #pred, a_nmse, gamma_1_log, gamma_2_log, required_res_norm_log, res_norm_log, input_RMSE, output_RMSE = pds_normal(
    #    masked_kspace, model, args, mri, target, 50, 400, sens_map_foo)

    if args.debug:
        plt.plot(a_nmse)
        plt.xlabel('iter')
        plt.ylabel('psnr')
        plt.grid()
        plt.savefig('test.png')
        
        #plt.imshow(pred, cmap='gray')
        #plt.savefig('pred.png')

        # x_bp = transforms.center_crop(transforms.complex_abs(mri.H(kspace)),(args.resolution, args.resolution)).cpu().numpy()
        # plt.imshow(x_bp, origin='lower', cmap='gray')
        # plt.title('backprojection')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

    pred = transforms.complex_abs(pred).cpu().numpy()
    exit()
    if args.optimal_scaling:
        pred, alpha = optimal_scale(target, pred, return_alpha=True)
        print(alpha)

    # Crop the predicted image to the correct size
    return pred


"""
rss_to_gt converts root sum of squares to true scaling

args:
    x_rss ((batch), y, x)
    sens_maps ((batch), coil, y, x) (complex)

"""
def gt_to_rss(x_gt, sens_maps):
    rss_map = np.sqrt(np.sum(np.abs(sens_maps)**2,axis=-3))
    return x_gt*rss_map


def admm(y, model, args, target):
    with torch.no_grad():
        # scale kspace
        if args.normalize == 'kspace':
            k_scale = torch.sqrt(torch.sum(y ** 2))
            y = y / k_scale
            target = target / k_scale.cpu().numpy()
            # scale beta parameter
            beta = args.beta / (k_scale ** 2)
        else:
            beta = args.beta

        Ht_y = fastmri.ifft2c(y)
        x = Ht_y * 0
        v = x
        u = x * 0

        outer_iters = 50
        inner_iters = args.inner_iters

        pnp = True
        inner_denoiser_iters = args.inner_denoiser_iters

        a_nmse = []

        # pl.ImagePlot(transforms.center_crop(transforms.complex_abs(Ht_y), (args.resolution, args.resolution)).cpu().numpy())

        for k in range(outer_iters):

            # Part1 of the ADMM, approximates the solution of:
            # x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2

            for j in range(inner_iters):
                b = Ht_y + beta * (v - u)
                A_x_est = fastmri.ifft2c(fastmri.fft2c(x)) + beta * x
                res = b - A_x_est
                a_res = fastmri.ifft2c(fastmri.fft2c(res)) + beta * res
                mu_opt = torch.mean(res * res) / torch.mean(res * a_res)
                # num = torch.sum(transforms.complex_mult(res, transforms.complex_conj(res))[..., 0])
                # den = torch.sum(transforms.complex_mult(a_res, transforms.complex_conj(res))[..., 0])
                # mu_opt = num / den
                x = x + mu_opt * res

            if pnp:
                v = denoiser(x + u, model, args)
            else:
                # Part2 of the ADMM, approximates the solution of
                # v = argmin_z lambda*z'*(z-denoiser(z)) +  0.5*beta||z - x - u||_2^2
                # using gradient descent
                for j in range(inner_denoiser_iters):
                    f_v = denoiser(v, model, args)
                    v = (beta * (x + u) + args.lamda * f_v) / (args.lamda + beta)

            # Part3 of the ADMM, update the dual variable
            u = u + x - v

            # Find psnr at every iteration
            x_crop = transforms.center_crop(transforms.complex_abs(x), (320, 320))
            nmse_step = psnr_2(transforms.complex_abs(target), x_crop)
            print(nmse_step)
            a_nmse.append(nmse_step)
        # unnormalize
        if args.normalize == 'kspace':
            x = x * k_scale
    return x, a_nmse

def main(args):
    # with multiprocessing.Pool(20) as pool:
    #     start_time = time.perf_counter()
    #     outputs = pool.map(run_model, range(len(data)))
    #     time_taken = time.perf_counter() - start_time
    #     logging.info(f'Run Time = {time_taken:}s')
    #     save_outputs(outputs, args.output_path)

    # handle pytorch device
    device = torch.device("cuda")

    # load model
    if args.checkpoint is not None:
        if args.natural_image:
            model = torch.load(args.checkpoint)
        else:
            _, model, _ = load_model(args.checkpoint)
        model.to(device)
        model.eval()
    else:
        model = None

    # non pooled version
    start_time = time.perf_counter()
    outputs = []
    a_metrics = []


    if args.debug:
        test_array = [args.test_idx]
    else:
        test_array = range(len(data))

    for i in test_array:
        # print('Test ' +str(i)+ ' of ' + str(len(test_array)), end='\r')
        print('Test ' +str(i)+ ' of ' + str(len(test_array)))
        masked_kspace, mask, sens, target, fname, slice = data[i]
        prediction = cs_pnp(args, model, masked_kspace, mask, sens, target)
        outputs.append( [fname, slice, prediction] )
        # compute metrics 
        NMSE = nmse(target, prediction)
        rSNR = 10*np.log10(1/NMSE)
        #PSNR = psnr(target, prediction)
        pred_clipped = prediction/np.max(prediction)
        metrics = [NMSE, rSNR]
        a_metrics.append(metrics)
        print('NMSE: {0:.4g}'.format(NMSE))

    time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    # Print metrics
    a_metrics = np.array(a_metrics)
    a_names = ['NMSE','rSNR']
    mean_metrics = np.mean(a_metrics, axis=0)
    std_metrics = np.std(a_metrics, axis=0)
    
    for i in range(len(a_names)):
        print(a_names[i] + ': ' + str(mean_metrics[i]) + ' +/- '+ str(2*std_metrics[i]))

    save_outputs(outputs, args.output_path)

    if args.output_path is not None:
        metric_file = args.output_path / 'metrics'
        np.save(metric_file, a_metrics)

def save_outputs(outputs, output_path):
    if output_path is None:
        return
    reconstructions = defaultdict(list)
    times = defaultdict(list)
    for fname, slice, pred in outputs:
        reconstructions[fname].append((slice, pred))
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    utils.save_reconstructions(reconstructions, output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, default='/storage/fastMRI_brain/data/Matt_preprocessed_data/T2')
    parser.add_argument('--output-path', type=pathlib.Path, default='out',
                        help='Path to save the reconstructions to')
    parser.add_argument('--snr', type=float, default=None, help='measurement noise')
    parser.add_argument('--project', default=False, action='store_true', help='replace loss prox with projection operator')
    parser.add_argument('--algorithm', type=str, default='pnp-admm', help='Algorithm used (pnp-pg, pnp-admm, red-admm, red-gd)')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--inner-iters', type=int, default=3,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--inner-denoiser-iters', type=int, default=1,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--step-size', type=float, default=None,
                        help='Step size parameter')
    parser.add_argument('--lamda', type=float, default=0.01, help='Regularization weight parameter')
    parser.add_argument('--relaxation', type=float, default=0.000, help='Relaxation of denoiser in PnP-PG')
    parser.add_argument('--beta', type=float, default=0.0005, help='ADMM Penalty parameter')
    parser.add_argument('--device', type=int, default=0, help='Cuda device (-1 for CPU)')
    parser.add_argument('--denoiser-mode', type=str, default='2-chan', help='Denoiser mode (mag, real_imag, 2-chan)')
    parser.add_argument('--checkpoint', type=str, default='/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP/best_model.pt', help='Path to an existing checkpoint.')
    parser.add_argument("--debug", default=True, action="store_true" , help="Debug mode")
    parser.add_argument("--test-idx", type=int, default=1, help="test index image for debug mode")
    parser.add_argument("--natural-image", default=False, action="store_true", help="Uses a pretrained DnCNN rather than a custom trained network")
    parser.add_argument("--normalize", type=str, default=None, help="Type of normalization going into denoiser (None, 'max', 'std')")
    parser.add_argument('--rotation-angles', type=int, default=0, help='number of rotation angles to try (<1 gives no rotation)')
    parser.add_argument("--accel", default=False, action='store_true', help="apply nesterov acceleration")
    parser.add_argument("--use-mid-slices", default=False, action='store_true', help="use only middle slices")
    parser.add_argument("--scanner-strength", type=float, default=None, help="Leave as None for all, >2.2 for 3, > 2.2 for 1.5")
    parser.add_argument('--mask-path', type=str, default=None, help='Path to mask (saved as Tensor)')
    parser.add_argument('--nc', type=int, default=4, help='number of coils to simulate')
    parser.add_argument('--coil-root', type=str, default='/home/reehorst.3/Documents/Reehorst_coil_maps/', help='path to coil directory')
    parser.add_argument("--scanner-mode", type=str, default=None, help="Leave as None for all, other options are PD, PDFS")
    parser.add_argument("--espirit-cal", default=False, action="store_true", help="set to use espririt calibrated maps")
    parser.add_argument('--run-name', default=None, type=str, help='wandb run name')
    parser.add_argument('--rss-target', default=False, action='store_true', help="Use rss as target (otherwise use gt)")
    parser.add_argument('--optimal-scaling', default=False, action='store_true', help="Optimal scaling")
    args = parser.parse_args()

    # restrict visible cuda devices
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    data = create_data_loader(args)
    main(args)
