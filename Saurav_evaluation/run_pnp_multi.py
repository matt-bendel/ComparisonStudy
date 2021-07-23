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

from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


# import bart
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

from common import utils
from common.args import Args
from common.subsample import MaskFunc
from common.utils import tensor_to_complex_np
from common.evaluate import nmse, psnr
from skimage.measure import compare_ssim

from data import transforms
from data.mri_data import SelectiveSliceData
from data.multicoil_sim import get_coil_images, mask_coil_images

from models.PnP.dncnn import DnCNN
from models.PnP.train_denoiser_mid import load_model

from models.unet.run_unet_multi import optimal_scale

import wandb

import scipy.misc
import PIL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def __call__(self, kspace, target, attrs, fname, slice):
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
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # generate coil images
        maps = 100
        if args.debug:
            maps= 1
        coil_images, sens = get_coil_images(self.args, kspace, total_maps=maps)

        # if args.debug:
        #     sens_np = tensor_to_complex_np(coil_images)
        #     print(sens_np.shape)
        #     pl.ImagePlot(sens_np, z=0)

        # generate target from coil images
        if args.rss_target:
            target = transforms.complex_center_crop(coil_images, (args.resolution, args.resolution))
            target = transforms.complex_abs(target)
            target = transforms.root_sum_of_squares(target)
            target = target.cpu().numpy()

        # subsample measurments
        masked_kspace, mask = mask_coil_images(self.args, coil_images, mask=self.mask)

        return masked_kspace, mask, sens, target, fname, slice

def create_data_loader(args):
    # select subset
    use_mid_slices=args.use_mid_slices
    fat_supress=None
    strength_3T=None

    if args.scanner_mode is not None:
        fat_supress = (args.scanner_mode == "PDFS")
    if args.scanner_strength is not None:
        strength_3T = (args.scanner_strength >=2.2)

    dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    data_set = SelectiveSliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(args),
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        use_mid_slices=use_mid_slices,
        fat_supress=fat_supress,
        strength_3T=strength_3T,
        restrict_size=True,
    )
    return data_set

def denoiser(noisy,model,args):
    # add rotate
    noisy, rot_angle = transforms.best_rotate(noisy, args.rotation_angles)

    # normalize
    if (args.normalize == 'max') or (args.normalize == 'std'):
        noisy, scale = transforms.denoiser_normalize(noisy, is_complex=True, use_std=args.normalize=='std')
    elif args.normalize=='constant':
        scale = 0.0012
        if args.espirit_cal:
            scale = scale/15
        noisy = noisy*(1/scale)
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
        noisy = noisy.permute(2,0,1).unsqueeze(0)
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
    
def admm(y, mri, model, args, target):
    with torch.no_grad():
        # scale kspace
        if args.normalize == 'kspace':
            k_scale = torch.sqrt(torch.sum(y**2))
            y = y/k_scale
            target = target/k_scale.cpu().numpy()
            # scale beta parameter
            beta = args.beta / (k_scale**2)
        else:
            beta = args.beta

        Ht_y = mri.H(y)
        x = Ht_y*0
        v = x
        u = x*0
        
        outer_iters = args.num_iters
        inner_iters = args.inner_iters
        
        pnp = (args.algorithm == 'pnp-admm')
        inner_denoiser_iters = args.inner_denoiser_iters
        
        a_nmse = []


        # pl.ImagePlot(transforms.center_crop(transforms.complex_abs(Ht_y), (args.resolution, args.resolution)).cpu().numpy())

        for k in range(outer_iters):
    
        # Part1 of the ADMM, approximates the solution of:
        # x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2

            for j in range(inner_iters):
                b = Ht_y + beta*(v - u)
                A_x_est = mri.H(mri.A(x)) + beta*x
                res = b - A_x_est
                a_res = mri.H(mri.A(res)) + beta*res
                mu_opt = torch.mean(res*res)/torch.mean(res*a_res)
                #num = torch.sum(transforms.complex_mult(res, transforms.complex_conj(res))[..., 0])
                #den = torch.sum(transforms.complex_mult(a_res, transforms.complex_conj(res))[..., 0])
                #mu_opt = num / den
                x = x + mu_opt*res

            if pnp:
                v = denoiser(x + u, model, args)
            else:
                # Part2 of the ADMM, approximates the solution of
                # v = argmin_z lambda*z'*(z-denoiser(z)) +  0.5*beta||z - x - u||_2^2
                # using gradient descent
                for j in range(inner_denoiser_iters):
                    f_v = denoiser(v, model, args)
                    v = (beta*(x + u) + args.lamda*f_v)/(args.lamda + beta)
    
            # Part3 of the ADMM, update the dual variable
            u = u + x - v
        
            # Find psnr at every iteration
            x_crop = transforms.center_crop(transforms.complex_abs(x).cpu().numpy(), (args.resolution, args.resolution))
            nmse_step = nmse(target, x_crop)
            a_nmse.append(nmse_step)
        # unnormalize
        if args.normalize=='kspace':
            x = x*k_scale
    return x, a_nmse


def gd_opt(y, mri, args, target):
    with torch.no_grad():
        # scale kspace

        Ht_y = mri.H(y)
        x = Ht_y*0

        outer_iters = args.num_iters

        a_nmse = []
        a_loss = []

        for k in range(outer_iters):
            Ax = mri.A(x)
            a_loss.append(torch.sum((Ax - y)**2).item())
            b = Ht_y
            A_x_est = mri.H(Ax)
            res = b - A_x_est
            a_res = mri.H(mri.A(res))
            # mu_opt = torch.mean(res * res) / torch.mean(res * a_res)
            num = torch.sum(transforms.complex_mult(res, transforms.complex_conj(res))[...,0])
            den = torch.sum(transforms.complex_mult(a_res, transforms.complex_conj(res))[...,0])
            mu_opt = num / den
            x = x + mu_opt * res

            # Find psnr at every iteration
            x_crop = transforms.center_crop(transforms.complex_abs(x).cpu().numpy(), (args.resolution, args.resolution))
            if args.optimal_scaling:
                x_crop = optimal_scale(target, x_crop)
            nmse_step = nmse(target, x_crop)
            a_nmse.append(nmse_step)

        if args.debug:
            plt.loglog(a_loss)
            plt.show()
    return x, a_nmse


def dumb_gd(y, mri, args, target):
    with torch.no_grad():

        Ht_y = mri.H(y)
        x = Ht_y * 0

        y1 = x
        t1 = 1

        # Compute step size
        if args.step_size is None:
            # Power iteration to find L
            L = find_spec_rad(mri, 20, x)
            mu = 1 / L
            print(mu)
        else:
            mu = args.step_size

        outer_iters = args.num_iters

        a_nmse = []

        # pl.ImagePlot(transforms.center_crop(transforms.complex_abs(Ht_y), (args.resolution, args.resolution)).cpu().numpy())

        for k in range(outer_iters):
            t0 = t1
            # Part1 of the ADMM, approximates the solution of:
            # x = argmin_z 1/(2sigma^2)||Hz-y||_2^2 + 0.5*beta||z - v + u||_2^2
            y0 = y1
            y1 = x - mu*(mri.H(mri.A(x))- Ht_y)

            if args.accel:
                t1 = (1 + np.sqrt(1 + 4*t0**2))/2
            else:
                t1 = 1
            x = y1 + ((t0-1)/t1)* (y1 - y0)

            # Find psnr at every iteration
            x_crop = transforms.center_crop(transforms.complex_abs(x).cpu().numpy(), (args.resolution, args.resolution))
            nmse_step = nmse(target, x_crop)
            a_nmse.append(nmse_step)
    return x, a_nmse

def find_spec_rad(mri,steps, x):
    # init x
    x = torch.randn_like(x)
    x = x/torch.sqrt(torch.sum(torch.abs(x)**2))

    # power iteration
    for i in range(steps):
        x = mri.H(mri.A(x))
        spec_rad = torch.sqrt(torch.sum(torch.abs(x)**2))
        x = x/spec_rad


    return spec_rad

def pnp_pg(y,mri,model,args, target):
    with torch.no_grad():

        Ht_y = mri.H(y)
        x = Ht_y

        # Compute step size
        if args.step_size is None:
            # Power iteration to find L
            L = find_spec_rad(mri,20, x)
            mu = 2/L
        else:
            mu = args.step_size

        y1 = x
        t1 = 1
        a_nmse = []
        for k in range(args.num_iters):
            t0 = t1
            y0 = y1
            # perform gradient step
            # z = x - mu*mri.H(mri.A(x)-y)
            z = x - mu * (mri.H(mri.A(x)) - Ht_y)
            # perform denoising
            y1 = (1-args.relaxation)*denoiser(z, model, args)+args.relaxation*z
            if args.accel:
                t1 = (1 + np.sqrt(1 + 4*t0**2))/2
            else:
                t1 = 1
            x = y1 + ((t0-1)/t1)* (y1 - y0)
            # x_crop = transforms.center_crop(transforms.complex_abs(x).cpu().numpy(), (args.resolution, args.resolution))
            z_crop = transforms.center_crop(transforms.complex_abs(z).cpu().numpy(), (args.resolution, args.resolution))
            # nmse_step = nmse(target, x_crop)
            nmse_step = nmse(target, z_crop)
            a_nmse.append(nmse_step)
        return x, a_nmse

def red_gd(y,mri,model,args, target):
    with torch.no_grad():
        x = mri.H(y)
        y1 = x
        t1 = 1
        a_nmse = []
        for k in range(args.num_iters):
            t0 = t1
            y0 = y1
            # perform denoising
            fx = denoiser(x, model, args)
            # perform gradient step
            y1 = x - args.step_size*(mri.H(mri.A(x)-y) + args.lamda*(x-fx))
            if args.accel:
                t1 = (1 + np.sqrt(1 + 4*t0**2))/2
            else:
                t1 = 1
            x = y1 + ((t0-1)/t1)* (y1 - y0)
            x_crop = transforms.center_crop(transforms.complex_abs(x).cpu().numpy(), (args.resolution, args.resolution))
            nmse_step = nmse(target, x_crop)
            a_nmse.append(nmse_step)
        return x, a_nmse

class A_mri:
    def __init__(self,sens_maps,mask):
        self.sens_maps = sens_maps
        self.mask = mask
    
    def A(self,x):
        x = x[None, ...]
        y = transforms.complex_mult(x,self.sens_maps)
        y_fft = transforms.fft2(y)
        out = self.mask * y_fft
        return out

    def H(self,x):
        y = self.mask*x
        y_ifft = transforms.ifft2(y)
        out = torch.sum(transforms.complex_mult(y_ifft,transforms.complex_conj(self.sens_maps)), dim=0)
        return out
        

def cs_pnp(args, model, kspace, mask, sens, target):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """
    # mask = mask.permute(0,2,1)
    kspace_np = tensor_to_complex_np(kspace)
    # mask = mask.cpu().numpy()

    if args.espirit_cal:
        # call Espirit
        device=sp.Device(0)
        sens_maps = mr.app.EspiritCalib(kspace_np,device=device).run()
        sens_maps = sp.to_device(sens_maps, -1)
        if args.debug:
            real_sens_maps_pwr = np.sum(np.abs(np.real(sens_maps))**2)
            imag_sens_maps_pwr = np.sum(np.abs(np.imag(sens_maps))**2)
            print("sens maps power (real/imag): {} / {}".format(real_sens_maps_pwr, imag_sens_maps_pwr))
        sens_maps = transforms.to_tensor(sens_maps.astype('complex64'))
    else:
        sens_maps = sens

    # handle pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sens_maps = sens_maps.to(device)
    mask = mask.to(device)

    mri = A_mri(sens_maps, mask)
    
    # Use PnP-PG to reconstruct the image
    kspace = kspace.to(device)

    if args.algorithm == 'pnp-pg':
        pred, a_nmse = pnp_pg(kspace, mri, model, args, target)

    elif args.algorithm == 'red-gd':
        pred, a_nmse = red_gd(kspace, mri, model, args, target)
        
    elif (args.algorithm == 'red-admm') or (args.algorithm == 'pnp-admm'):
        pred, a_nmse = admm(kspace, mri, model, args, target)
    elif (args.algorithm =='dumb-gd'):
        pred, a_nmse = dumb_gd(kspace,mri,args, target)
    elif (args.algorithm == 'gd-opt'):
        pred, a_nmse = gd_opt(kspace, mri, args, target)




    if args.debug:
        plt.loglog(range(args.num_iters), a_nmse)
        plt.xlabel('iter')
        plt.ylabel('nmse')
        plt.grid()
        plt.show()

        # x_bp = transforms.center_crop(transforms.complex_abs(mri.H(kspace)),(args.resolution, args.resolution)).cpu().numpy()
        # plt.imshow(x_bp, origin='lower', cmap='gray')
        # plt.title('backprojection')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()

    pred = transforms.complex_abs(pred).cpu().numpy()

    pred = transforms.center_crop(pred, (args.resolution, args.resolution))

    if args.rss_target and (not args.espirit_cal):
        sens_np = tensor_to_complex_np(transforms.complex_center_crop(sens_maps, (args.resolution, args.resolution)).cpu())
        pred = gt_to_rss(pred, sens_np)

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

def main(args):
    # with multiprocessing.Pool(20) as pool:
    #     start_time = time.perf_counter()
    #     outputs = pool.map(run_model, range(len(data)))
    #     time_taken = time.perf_counter() - start_time
    #     logging.info(f'Run Time = {time_taken:}s')
    #     save_outputs(outputs, args.output_path)

    # handle pytorch device
    wandb.init(config=args, project='fastmri-multi-pnp', name=args.run_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        PSNR = psnr(target, prediction)
        SSIM = compare_ssim(target, prediction, data_range=target.max())
        pred_clipped = prediction/np.max(prediction)
        wandb.log({'PSNR': PSNR,
                   'NMSE': NMSE,
                   'SSIM': SSIM,
                   'rSNR': rSNR,
                   'pred': wandb.Image(pred_clipped)})
        metrics = [NMSE, PSNR, SSIM, rSNR]
        a_metrics.append(metrics)
        print('NMSE: {0:.4g}'.format(NMSE))
        if args.debug:
            # display = np.concatenate([prediction, target], axis=1)
            # pl.ImagePlot(display)
            plt.imshow(prediction, origin='lower', cmap='gray')
            plt.title('PnP reconstruction')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            plt.imshow(target, origin='lower', cmap='gray')
            plt.title('target')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            error = np.abs(target-prediction)

            plt.imshow(error, origin='lower', cmap='gray')
            plt.title('error')
            plt.xticks([])
            plt.yticks([])
            plt.show()

            if True:
                scipy.io.savemat('reconstruction.mat', {'recon': prediction})
                scipy.io.savemat('target.mat', {'target':target})
                scipy.io.savemat('error.mat', {'error':error})
                PIL.ImageOps.flip(scipy.misc.toimage(prediction, cmin = 0.0, cmax = np.max(target))).save('pred.eps')
                PIL.ImageOps.flip(scipy.misc.toimage(target, cmin = 0.0, cmax = np.max(target))).save('target.eps')
                PIL.ImageOps.flip(scipy.misc.toimage(error*4, cmin = 0.0, cmax = np.max(target))).save('errorx4.eps')
    time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    # Print metrics
    a_metrics = np.array(a_metrics)
    a_names = ['NMSE','PSNR','SSIM','rSNR']
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
    for fname, slice, pred in outputs:
        reconstructions[fname].append((slice, pred))
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    utils.save_reconstructions(reconstructions, output_path)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--output-path', type=pathlib.Path, default=None,
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
    parser.add_argument('--beta', type=float, default=0.001, help='ADMM Penalty parameter')
    parser.add_argument('--device', type=int, default=0, help='Cuda device (-1 for CPU)')
    parser.add_argument('--denoiser-mode', type=str, default='2-chan', help='Denoiser mode (mag, real_imag, 2-chan)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to an existing checkpoint.')
    parser.add_argument("--debug", default=False, action="store_true" , help="Debug mode")
    parser.add_argument("--test-idx", type=int, default=0, help="test index image for debug mode")
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = create_data_loader(args)
    main(args)
