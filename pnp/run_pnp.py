# Author: Saurav
# This file contains the PnP reconstruction and Baseline U-Net reconstruction results
# Specifications:

# Brain Data: /storage/fastMRI_brain/data/
# GRO Sampling Pattern
# acceleration R = 4
# No noise added

# Training Data: (refer notability notes) It uses 1541 (3T AXT2 volumes with coils >= 8) x 8 (bottom large brain slices) = 12328 image data from /storage/fastMRI_brain/data/multicoil_train
# Validation Data: (refer notability notes) It uses 421 (3T AXT2 volumes with coils >= 8) x 8 (bottom large brain slices) = 3368 image data from /storage/fastMRI_brain/data/multicoil_val
# Testing Data is same as Validation Data (might have considered subset of Validation for testing)
# the above numbers might be a rough estimate since a few images are left out because of size restiction and number of coil . I think the numbers are 12256 and 3352

# The multicoil data (coils>=8) was reduced to 8 coil data using the compression procedure from Prof. Ahmad's group. Prof. Ahmad mentioned that this is a standard procedure and we donot loose any significant information during this step.

# Both PnP and U-Net code reconstructs 384x384 image, but for computing performance we look at center 320x320 image.


######## IMPORTANT #########

# The code requires you to modify the spectral_nrom.py file

######## IMPORTANT #########

import numpy as np
import os
import torch
from espirit import ifft, fft
import sigpy as sp
import sigpy.mri as mr
from utils.fastmri import tensor_to_complex_np
from pnp import transforms
from utils.fastmri.data.mri_data import SelectiveSliceData_Val
from utils.fastmri.models.PnP.train_denoiser import load_model
from utils.fastmri import utils
from collections import defaultdict
import logging
import time
from eval import nmse, psnr
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import pickle
import pathlib

class Init_Arg:
    def __init__(self):
        self.resolution = 320
        self.test_idx = 2
        self.data_path_val = '/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_test'
        self.sample_rate = 1.
        self.accelerations = [4]
        self.center_fractions = [0.08]
        self.output_path = 'out'
        self.algorithm = 'pnp-pds'
        self.num_iters = 100
        self.beta = 0.0005  # 0.0005 works well, 0.001 gives okish results. 0.001 works good for admm # Not used
        self.device = 0
        self.denoiser_mode = '2-chan'
        self.checkpoint_denoiser = '/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP/checkpoints/best_model_30db.pt'
        self.normalize = 'constant'
        self.rotation_angles = 0
        self.accel = False
        self.nc = 1
        self.rss_target = True
        self.num_of_top_slices = 8
        self.debug = False
        self.snr = 20
        self.output_path = pathlib.Path('out')


def flatten(t):
    t = t.reshape(1,-1)
    t = t.squeeze()
    return t

def unflatten(t,shape_t):
    t = t.reshape(shape_t)
    return t

def nmse_tensor(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return torch.norm(gt - pred) ** 2 / torch.norm(gt) ** 2

def psnr_tensor(gt, pred):
    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args, use_seed=False):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.use_seed = use_seed
        self.args = args
        self.mask = None

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # GRO Sampling mask:
        a = np.array([0,8,16,24,31,39,45,52,58,64,70,76,81,87,92,96,101,105,110,114,118,122,126,130,133,137,140,144,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,173,176,180,183,187,190,194,198,202,206,210,215,219,224,228,233,239,244,250,256,262,268,275,281,289,296,304,312])

        m = np.zeros((320, 320))
        m[:, a] = True
        m[:, 176:208] = True

        samp = m
        numcoil = 1
        mask = np.tile(samp, (numcoil, 1, 1)).transpose((1, 2, 0)).astype(np.float32)

        kspace = np.expand_dims(kspace, axis=0)
        kspace = kspace.transpose(1, 2, 0)

        x = ifft(kspace, (0, 1))  # (320, 320, 1)

        RSS_x = np.squeeze(x)  # (320, 320)

        kspace = fft(x, (1, 0))  # (320, 320, 1)

        masked_kspace = kspace * mask

        kspace = transforms.to_tensor(kspace)
        kspace = kspace.permute(2, 0, 1, 3)

        masked_kspace = transforms.to_tensor(masked_kspace)
        masked_kspace = masked_kspace.permute(2, 0, 1, 3)

        nnz_index_mask = mask[0, :, 0].nonzero()[0]

        sig_power = torch.sum(masked_kspace ** 2)
        snr_ratio = 10 ** (args.snr / 10)
        noise_var = sig_power / snr_ratio / (masked_kspace.shape[0] * masked_kspace.shape[1] * len(nnz_index_mask))

        #print('noise variance of this run:')
        #print(format(noise_var))


        nnz_masked_kspace = masked_kspace[:, :, nnz_index_mask, :]
        nnz_masked_kspace_real = nnz_masked_kspace[:, :, :, 0]
        nnz_masked_kspace_imag = nnz_masked_kspace[:, :, :, 1]
        nnz_masked_kspace_real_flat = flatten(nnz_masked_kspace_real)
        nnz_masked_kspace_imag_flat = flatten(nnz_masked_kspace_imag)

        noise_flat_1 = (torch.sqrt(0.5 * noise_var)) * torch.randn(nnz_masked_kspace_real_flat.size())
        noise_flat_2 = (torch.sqrt(0.5 * noise_var)) * torch.randn(nnz_masked_kspace_real_flat.size())

        nnz_masked_kspace_real_flat_noisy = nnz_masked_kspace_real_flat.float() + noise_flat_1.float()
        nnz_masked_kspace_imag_flat_noisy = nnz_masked_kspace_imag_flat.float() + noise_flat_2.float()

        nnz_masked_kspace_real_noisy = unflatten(nnz_masked_kspace_real_flat_noisy, nnz_masked_kspace_real.shape)
        nnz_masked_kspace_imag_noisy = unflatten(nnz_masked_kspace_imag_flat_noisy, nnz_masked_kspace_imag.shape)

        nnz_masked_kspace_noisy = nnz_masked_kspace * 0
        nnz_masked_kspace_noisy[:, :, :, 0] = nnz_masked_kspace_real_noisy
        nnz_masked_kspace_noisy[:, :, :, 1] = nnz_masked_kspace_imag_noisy

        masked_kspace_noisy = 0 * masked_kspace
        masked_kspace_noisy[:, :, nnz_index_mask, :] = nnz_masked_kspace_noisy

        # testing
        pow_1 = torch.sum(nnz_masked_kspace_real_flat ** 2)
        pow_2 = torch.sum(nnz_masked_kspace_imag_flat ** 2)
        pow_3 = torch.sum(noise_flat_1 ** 2)
        pow_4 = torch.sum(noise_flat_2 ** 2)
        ratio_snr = torch.sqrt(pow_1 + pow_2) / torch.sqrt(pow_3 + pow_4)
        SNRdB_test = 20 * torch.log10(ratio_snr)
        #print('SNR in dB for this run:')
        #print(SNRdB_test)

        masked_kspace = masked_kspace_noisy

        args.resnorm = torch.sqrt(noise_var * (masked_kspace.shape[0] * masked_kspace.shape[1] * len(nnz_index_mask)))
        args.resnorm_2 = torch.sqrt(noise_var * (masked_kspace.shape[0] * masked_kspace.shape[1] * len(nnz_index_mask)))
        args.noise_var = noise_var

        kspace_np = tensor_to_complex_np(masked_kspace)
        ESPIRiT_width_mask = 24
        device = sp.Device(0)

        sens_maps = mr.app.EspiritCalib(kspace_np, calib_width=ESPIRiT_width_mask, device=device, show_pbar=False).run()

        sens_maps = sp.to_device(sens_maps, -1)
        sens_map_foo = np.zeros((args.resolution, args.resolution, 1)).astype(np.complex)
        sens_map_foo[:, :, 0] = sens_maps[0, :, :]

        lsq_gt = np.sum(sens_map_foo.conj() * x, axis=-1)

        return masked_kspace, mask, kspace, RSS_x, x, fname, slice, sens_map_foo, lsq_gt

def create_data_loader(args):
    train_data = SelectiveSliceData_Val(
        root=args.data_path_val,
        transform=DataTransform(args),
        challenge='singlecoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )

    return train_data

def Rss(x):
    y = np.expand_dims(np.sum(np.abs(x) ** 2, axis=-1) ** 0.5, axis=2)
    return y

def denoiser(noisy, model, args):
    # add rotate
    noisy, rot_angle = transforms.best_rotate(noisy, args.rotation_angles)

    # normalize
    if (args.normalize == 'max') or (args.normalize == 'std'):
        noisy, scale = transforms.denoiser_normalize(noisy, is_complex=True, use_std=args.normalize == 'std')
    elif args.normalize == 'constant':
        #         print('hi')
        scale = 0.0016
        #         if args.espirit_cal:
        #             scale = scale/15
        noisy = noisy * (1 / scale)
    else:
        scale = 1

    if args.denoiser_mode == 'mag':
        mag = transforms.complex_abs(noisy)
        phase = transforms.phase(noisy)
        denoised_mag = model(mag[None, None, ...])
        denoised_mag = denoised_mag[0, 0, ...]
        denoised_image = transforms.polar_to_rect(denoised_mag, phase)

    elif args.denoiser_mode == '2-chan':
        # move real/imag to channel position
        noisy = noisy.permute(2, 0, 1).unsqueeze(0)
        denoised_image = model(noisy).squeeze(0).permute(1, 2, 0)

    elif args.denoiser_mode == 'real-imag':
        # move real/imag to batch position
        noisy = noisy.permute(2, 0, 1).unsqueeze(1)
        denoised_image = model(noisy).squeeze(1).permute(1, 2, 0)

    # unnormalize

    denoised_image = denoised_image * scale

    # undo rotate
    denoised_image = transforms.polar_to_rect(transforms.complex_abs(denoised_image),
                                              transforms.phase(denoised_image) - rot_angle)

    return denoised_image

class A_mri:
    def __init__(self, sens_maps, mask):
        self.sens_maps = sens_maps
        self.mask = mask

    def A(self, x):
        x = x[None, ...]
        y = transforms.complex_mult(x, self.sens_maps)
        y_fft = transforms.fft2(y)
        out = self.mask * y_fft
        return out

    def H(self, x):
        y = self.mask * x
        y_ifft = transforms.ifft2(y)
        out = torch.sum(transforms.complex_mult(y_ifft, transforms.complex_conj(self.sens_maps)), dim=0)
        return out

class B_mri:
    def __init__(self, sens_maps, mask):
        self.sens_maps = sens_maps
        self.mask = mask

    def A(self, x):
        x = x[None, ...]
        y = transforms.complex_mult(x, self.sens_maps)
        y_fft = transforms.fft2(y)
        out = self.mask * y_fft
        return out

    def H(self, x):
        y = self.mask * x
        y_ifft = transforms.ifft2(y)
        out = torch.sum(transforms.complex_mult(y_ifft, transforms.complex_conj(self.sens_maps)), dim=0)
        return out

class C_mri:
    def __init__(self, mat_E, mri_B):
        self.mat_E = mat_E
        self.mri_B = mri_B

    def A(self, x):
        out = self.mri_B.H(self.mat_E * self.mri_B.A(x))
        return out

    def H(self, x):
        out = self.mri_B.H(self.mat_E * self.mri_B.A(x))
        return out

class D_mri:
    def __init__(self, sens_maps):
        self.sens_maps = sens_maps

    def A(self, x):
        x = x[None, ...]
        y = transforms.complex_mult(x, self.sens_maps)
        y_fft = transforms.fft2(y)
        out = y_fft
        return out

    def H(self, x):
        y = x
        y_ifft = transforms.ifft2(y)
        out = torch.sum(transforms.complex_mult(y_ifft, transforms.complex_conj(self.sens_maps)), dim=0)
        return out

def find_spec_rad(mri, steps, x):
    # init x
    x = torch.randn_like(x)
    x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2))

    # power iteration
    for i in range(steps):
        x = mri.H(mri.A(x))
        spec_rad = torch.sqrt(torch.sum(torch.abs(x) ** 2))
        x = x / spec_rad

    return spec_rad

def pds_normal(y, model, args, mri, target, max_iter, gamma_1_input, mri_B, mri_D, mask_tensor,
               mask_tensor_compliment, zeta, sens_map_foo):

    #     print('Running generic-2 PnP-PDS')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EYE = torch.ones(1, args.image_size, args.image_size, 2)
    EYE = EYE.to(device)

    sens_map_zero_count = 0
    AA = torch.zeros(args.image_size, args.image_size)
    AA_ones = torch.zeros(args.image_size, args.image_size) + 1
    for i in range(args.image_size):
        for j in range(args.image_size):
            if np.sum(np.abs(sens_map_foo[i, j, :])) == 0:
                sens_map_zero_count = sens_map_zero_count + 1
                AA[i, j] = 1
                AA_ones[i, j] = 0
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

        x = 1 * mri.H(y)
        z = 0 * x

        x_crop = transforms.complex_abs(x)
        #fig = plt.figure()
        #plt.imshow(np.abs(x_crop.cpu().numpy()),cmap='gray')
        #plt.savefig('pre_alg.png')
        nmse_step = nmse_tensor(target, x_crop)
        psnr_step = psnr_tensor(target, x_crop)
        a_nmse.append(nmse_step)
        a_rSNR.append(psnr_step)

        L = find_spec_rad(mri, 100, x)

        gamma_1 = gamma_1_input * roo

        gamma_2 = (1 / L) * ((1 / gamma_1))

        for k in range(outer_iters):
            yoda = 1
            b1 = x - (gamma_1 * mri.H(z))

            x_new = yoda * denoiser(b1, model, args) + (1 - yoda) * x

            x_hat = x_new + 1 * (x_new - x)

            z = (EYE - (1 / ((((1 / roo) * EYE) / gamma_2) + EYE))) * z + (
                        1 / ((((1 / roo) * EYE) / gamma_2) + EYE)) * ((1 / roo) * EYE) * (mri.A(x_hat) - y)

            x = x_new

            gamma_1_log.append(gamma_1)
            gamma_2_log.append(gamma_2)

            boo = mri.A(x) - y

            resnorm_recov = torch.sqrt(torch.sum(boo ** 2))

            x_crop = transforms.complex_abs(x)

            target_2 = target

            target_3 = target_2 * AA_ones
            x_crop_3 = x_crop * AA_ones
            rSNR_step = (1 / nmse_tensor(target_3, x_crop_3))
            nmse_step = nmse_tensor(target_3, x_crop_3)
            psnr_step = psnr_tensor(target_3, x_crop_3)

            a_nmse.append(nmse_step)
            a_rSNR.append(psnr_step)

            res_norm_log.append(resnorm_recov)

    return x, a_nmse, a_rSNR, gamma_1_log, gamma_2_log, required_res_norm_log, res_norm_log, input_RMSE, output_RMSE


def cs_pnp(args, model, masked_kspace, mask, kspace, RSS_x, x, fname, slice, sens_map_foo, lsq_gt):
    """
    Run ESPIRIT coil sensitivity estimation and Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """
    args.snr = 30
    args.add_noise = False
    args.image_size = 320

    sens_maps = np.zeros((1, 320, 320)).astype(np.complex)
    sens_maps = sp.to_device(sens_maps, -1)

    sens_maps[0, :, :] = sens_map_foo[:, :, 0]
    sens_maps = transforms.to_tensor(sens_maps.astype('complex64'))

    device = torch.device("cuda")
    sens_maps = sens_maps.to(device)

    mask_tensor = np.zeros((1, 320, 1, 1))
    mask_tensor[0, :, 0, 0] = mask[0, :, 0]
    mask_tensor = transforms.to_tensor(mask_tensor)
    mask_tensor = mask_tensor.permute(2, 0, 1, 3).float()

    mask_tensor = mask_tensor.to(device)
    masked_kspace_foo = masked_kspace.float().to(device)

    target = transforms.complex_abs(transforms.to_tensor(RSS_x)).float()
    target = target.to(device)

    mri = A_mri(sens_maps, mask_tensor)

    mask_tensor_compliment = torch.ones(mask_tensor.shape).to(device) - mask_tensor

    mri_B = B_mri(sens_maps, mask_tensor_compliment)
    mri_D = D_mri(sens_maps)

    max_iter = 50  # Requires early stopping. That's why max_iter limited to 50.

    zeta = 0.01
    gamma_inp = 400

    pred, a_nmse, a_PSNR, gamma_1_log, gamma_2_log, required_res_norm_log, res_norm_log, input_RMSE, output_RMSE = pds_normal(
        masked_kspace_foo, model, args, mri, target, max_iter, gamma_inp, mri_B, mri_D, mask_tensor,
        mask_tensor_compliment,
        zeta, sens_map_foo)

    if args.debug:
        fig = plt.figure()
        plt.plot(a_PSNR)
        plt.xlabel('iter')
        plt.ylabel('psnr')
        plt.grid()
        plt.savefig('test.png')


    pred = transforms.complex_abs(pred).cpu().numpy()
    #fig = plt.figure()
    #plt.imshow(np.abs(pred),cmap='gray')
    #plt.savefig('post_alg.png')
    return pred, target.cpu().numpy()

def main(args):
    # with multiprocessing.Pool(20) as pool:
    #     start_time = time.perf_counter()
    #     outputs = pool.map(run_model, range(len(data)))
    #     time_taken = time.perf_counter() - start_time
    #     logging.info(f'Run Time = {time_taken:}s')
    #     save_outputs(outputs, args.output_path)

    # handle pytorch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, model, _ = load_model(args.checkpoint_denoiser)

    model.to(device)
    model.eval()

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
        masked_kspace, mask, kspace, RSS_x, x, fname, slice, sens_map_foo, lsq_gt = data[i]
        print(fname)
        start_time_2 = time.perf_counter()
        prediction, target = cs_pnp(args, model, masked_kspace, mask, kspace, RSS_x, x, fname, slice, sens_map_foo, lsq_gt)
        time_taken_2 = time.perf_counter() - start_time_2
        #fig = plt.figure()
        #plt.imshow(np.abs(target),cmap='gray')
        #plt.savefig('gt.png')
        outputs.append( [fname, slice, prediction, time_taken_2] )

        # compute metrics
        NMSE = nmse(target, prediction)
        PSNR = psnr(target, prediction)
        rSNR = 10*np.log10(1/NMSE)
        metrics = [NMSE, rSNR]
        a_metrics.append(metrics)

        print(f'NMSE: {NMSE}\nPSNR: {PSNR}')

    time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken:}s')
    save_outputs(outputs, args.output_path)

def save_outputs(outputs, output_path):
    if output_path is None:
        return
    reconstructions = defaultdict(list)
    times = defaultdict(list)
    for fname, slice, pred, recon_time in outputs:
        reconstructions[fname].append((slice, pred))
        times[fname].append((slice, recon_time))
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }

    with open('out/recon_times.pkl', 'wb') as f:
        pickle.dump(times, f)

    utils.save_reconstructions(reconstructions, output_path)

if __name__ == '__main__':
    args = Init_Arg()

    args.use_pre_determined_noise_var = False
    args.add_noise = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    data = create_data_loader(args)

    main(args)

