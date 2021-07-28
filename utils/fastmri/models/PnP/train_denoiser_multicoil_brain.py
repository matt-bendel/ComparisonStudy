# Author: Saurav

# Code used for training the brain MRI multicoil denoiser

# Training Data: (refer notability notes) It uses 1541 (3T AXT2 volumes with coils >= 8) x 8 (bottom large brain slices) = 12328 image data from /storage/fastMRI_brain/data/multicoil_train
# Validation Data: (refer notability notes) It uses 421 (3T AXT2 volumes with coils >= 8) x 8 (bottom large brain slices) = 3368 image data from /storage/fastMRI_brain/data/multicoil_val

# the above numbers might be a rough estimate since a few images are left out because of size restiction and number of coil . I think the numbers are 12256 and 3352

# The coil sensitivites were generated offline and stored in '/storage/fastMRI_brain/coil_maps/' with the same file name as the data for both training and validation dataset

# old command used in order to run the program: PYTHONPATH=. python models/PnP/train_denoiser_multicoil_brain.py --snorm --L 0.76 --std 0.03 --data-parallel --run-name std0.03_Trail_1_brain

## Note this code requires you to set weight = weight / sigma*0.631 (in spectral_norm.py file in the python env) done in order to accurately control the spectral norm since controll "L" value doesnot really help
# new command which works for brain denoiser training with the above edit: PYTHONPATH=. python models/PnP/train_denoiser_multicoil_brain.py --snorm --L 1 --std 0.02 --data-parallel --run-name std0.02_Trail_6_brain

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time
import os

import numpy as np 
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader


from argparse import ArgumentParser
from utils import fastmri
from utils.fastmri.data import transforms

from utils.fastmri.data.mri_data import SelectiveSliceData_Train
from utils.fastmri.data.mri_data import SelectiveSliceData_Val
from utils.fastmri.data.mri_data import SliceDataset

from utils.fastmri.utils import generate_gro_mask

from utils.fastmri.models.unet.unet import UnetModel

from utils.fastmri.models.PnP.dncnn import DnCNN

import pathlib
from pathlib import Path 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gro_mask(mask_shape):
    #Get Saved CSV Mask
    mask = generate_gro_mask(mask_shape[-2])
    shape = np.array(mask_shape)
    shape[:-3] = 1
    num_cols = mask_shape[-2]
    mask_shape = [1 for _ in shape]
    mask_shape[-2] = num_cols
    return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

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

def Rss(x):
    y = np.expand_dims(np.sum(np.abs(x)**2,axis = -1)**0.5,axis = 2)
    return y

def ImageCropandKspaceCompression(x,image_size):
#     print(x.shape)
#     plt.imshow(np.abs(x[:,:,0]), origin='lower', cmap='gray')
#     plt.show()
        
    w_from = (x.shape[0] - image_size) // 2  # crop images into image_sizeximage_size
    h_from = (x.shape[1] - image_size) // 2
    w_to = w_from + image_size
    h_to = h_from + image_size
    cropped_x = x[w_from:w_to, h_from:h_to,:]
    
#     print('cropped_x shape: ',cropped_x.shape)
    if cropped_x.shape[-1] >= 8:
        x_tocompression = cropped_x.reshape(image_size**2,cropped_x.shape[-1])
        U,S,Vh = np.linalg.svd(x_tocompression,full_matrices=False)
        coil_compressed_x = np.matmul(x_tocompression, Vh.conj().T)
        coil_compressed_x = coil_compressed_x[:,0:8].reshape(image_size,image_size,8)
    else:
        coil_compressed_x = cropped_x
        
    return coil_compressed_x

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, noise_std, resolution=None, mag_only=False, normalize=None, rotation_angles=0,random_crop=True, rss_target=False, train_data=True, image_size = 320):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        # self.mask_func = mask_func
        self.resolution = resolution
        # self.which_challenge = which_challenge
        self.mag_only = mag_only
        self.normalize = normalize
        self.noise_std = noise_std
        self.num_angles = rotation_angles
        self.random_crop = random_crop
        self.rss_target = rss_target
        self.train = train_data
        self.image_size = image_size

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


#         device = torch.device(0)
        mask = get_gro_mask(kspace.shape)
        kspace = transforms.to_tensor(kspace)
        # kspace = (kspace * mask) + 0.0
        image = fastmri.ifft2c(kspace) #(320, 320)
        image, rot_angle = transforms.best_rotate(image, self.num_angles)

        scale = 0.0016 # constant scale
        image = image/scale

        scale = torch.tensor([scale], dtype=torch.float)

        # add noise
        target = image.float()
        # add zero mean noise
        image = image.float() + self.noise_std*torch.randn_like(image.float())

        #image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        #image = image.clamp(-6, 6)
        #
        #target = transforms.normalize(target, mean, std, eps=1e-11)
        #target = target.clamp(-6, 6)
        # move real/imag to channel position

        image = image.permute(2,0,1)
        target = target.permute(2,0,1)

        return image, target, scale, attrs['norm'].astype(np.float32), rot_angle


def create_datasets(args):

    train_data = SelectiveSliceData_Train(
        root=args.data_path / f'singlecoil_train',
        transform=DataTransform(args.std, args.patch_size, mag_only=args.denoiser_mode=='mag', normalize=args.normalize, rotation_angles=args.rotation_angles, random_crop=True, rss_target=args.rss_target, train_data=True, image_size = 320),
        challenge='singlecoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )

    dev_data = SelectiveSliceData_Val(
        root=args.data_path / f'singlecoil_val',
        transform=DataTransform(args.std, args.val_patch_size, mag_only=args.denoiser_mode=='mag', normalize=args.normalize, rotation_angles=args.rotation_angles, random_crop=False, rss_target=args.rss_target, train_data=False, image_size = 320),
        challenge='singlecoil',
        sample_rate=1,
        use_top_slices=True,
        number_of_top_slices=args.num_of_top_slices,
        fat_supress=None,
        strength_3T=None,
        restrict_size=False,
    )

    # train_data = SliceDataset(
    #     root=args.data_path / f'singlecoil_train',
    #     transform=DataTransform(args.std, args.patch_size, mag_only=args.denoiser_mode=='mag', normalize=args.normalize, rotation_angles=args.rotation_angles, random_crop=True, rss_target=args.rss_target, train_data=True, image_size = 384),
    #     sample_rate=1.0,
    #     challenge='singlecoil',
    # )
    # dev_data = SliceDataset(
    #     root=args.data_path / f'singlecoil_val',
    #     transform=DataTransform(args.std, args.val_patch_size, mag_only=args.denoiser_mode=='mag', normalize=args.normalize, rotation_angles=args.rotation_angles, random_crop=False, rss_target=args.rss_target, train_data=False, image_size = 384),
    #     sample_rate=1.0,
    #     challenge='singlecoil',
    # )
        
    return dev_data, train_data




def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    # display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
    )
    # display_loader = DataLoader(
    #     dataset=display_data,
    #     batch_size=16,
    #     num_workers=16,
    #     pin_memory=True,
    # )
    return train_loader, dev_loader
    # return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    losses = []
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, scale, norm, rot_angle = data
        input = input.cuda()
        target = target.cuda()
        if args.denoiser_mode == 'real-imag':
            input = torch.cat(torch.chunk(input, 2, dim=1), dim=0) # may need to unsqueeze
            target = torch.cat(torch.chunk(target, 2, dim=1), dim=0) # may need to unsqueeze
            scale = torch.cat([scale, scale], dim=0)

        output = model(input)
        if args.loss=='MSE':
            loss = F.mse_loss(output, target, reduction='sum')
        elif args.loss=='l1':
            loss = F.l1_loss(output, target, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    input_losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, scale, norm, rot_angle = data
            scale = scale[:, None, None, None].cuda()
            input = input.cuda()
            target = target.cuda()
            if args.denoiser_mode == 'real-imag':
                input = torch.cat(torch.chunk(input, 2, dim=1), dim=0) # may need to unsqueeze
                target = torch.cat(torch.chunk(target, 2, dim=1), dim=0) # may need to unsqueeze
                scale = torch.cat([scale, scale], dim=0)
            output = model(input)
            output_error = torch.sum(torch.abs(output-target)**2, dim=(1,2,3))
            input_error = torch.sum(torch.abs(input-target)**2, dim=(1,2,3))
            target_l2 = torch.sum(torch.abs(target)**2, dim=(1,2,3))
            input_nmse = (input_error/target_l2).tolist()
            output_nmse = (output_error/target_l2).tolist()
            losses = losses + output_nmse
            input_losses = input_losses + input_nmse
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag, args):
        if not ((args.denoiser_mode=='mag') or (args.denoiser_mode=='real-imag')):
            image = transforms.complex_abs(image.permute(0,2,3,1)).unsqueeze(3).permute(0,3,1,2)
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)
        grid_np = np.transpose(grid.cpu().numpy(), (1,2,0))

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, scale, norm, rot_angle = data
            input = input.cuda()
            target = target.cuda()
            if args.denoiser_mode == 'real-imag':
                input = torch.cat(torch.chunk(input, 2, dim=1), dim=0) # may need to unsqueeze
                target = torch.cat(torch.chunk(target, 2, dim=1), dim=0) # may need to unsqueeze
                scale = torch.cat([scale, scale], dim=0)
            output = model(input)
            save_image(input, 'input', args)
            save_image(target, 'Target', args)
            save_image(output, 'Reconstruction',args)
            save_image(target - output, 'Error',args)
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    if (args.denoiser_mode == 'mag') or (args.denoiser_mode == 'real-imag'):
        chans = 1
    else:
        chans = 2
    if args.denoiser == 'DnCNN':
        batch_norm = 'full'
        if args.snorm:
            batch_norm = 'mean'
        rsnorm = False
        if hasattr(args, 'realsnorm'):
            rsnorm = args.realsnorm
            if args.realsnorm:
                batch_norm = 'mean'
        residual=True
        if hasattr(args, 'direct'):
            residual = not args.direct
        model = DnCNN(
            depth=args.num_layers,
            image_channels=chans,
            n_channels=args.num_chans,
            snorm = args.snorm,
            realsnorm = rsnorm,
            L = args.L,
            bnorm_type = batch_norm,
            residual=residual,
        ).cuda()
    else:
        model = UnetModel(chans, chans, args.num_chans, args.num_pools, args.drop_prob).cuda()
    return model


def load_model(checkpoint_file):
    device = torch.device('cuda')
    checkpoint = torch.load(checkpoint_file, map_location=device)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('Resuming Training...')
        checkpoint, model, optimizer = load_model(args.checkpoint)
        print(" ")
        print(" ")
        print("Need to edit these below things.")
        print(" ")
        print(" ")
        args = checkpoint['args']
        args.lr = 1e-3
        args.exp_dir = Path('/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP/')
        args.run_name = 'denoiser_train'
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        # visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--num-layers', type=int, default=5, help='Number of dncnn layers') # Ted used 17 layers
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--denoiser', type=str, default='DnCNN')
    parser.add_argument('--direct', action='store_true', help='direct denoiser (instead of residual)')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--num-chans', type=int, default=64, help='Number of U-Net channels')
    parser.add_argument('--snorm', default=True, action='store_true', help='Turns on spectral normalization')
    parser.add_argument('--realsnorm', action='store_true', help='Turns on real spectral normalization')
    parser.add_argument('--L', type=float, default=1, help='Lipschitz constant of network')
    # parser.add_argument('--mag-only', action='store_true', help='denoise mag only')
    parser.add_argument('--denoiser-mode', type=str, default='2-chan')
    parser.add_argument('--rotation-angles', type=int, default=0, help='number of rotation angles to try (<1 gives no rotation)')
    parser.add_argument('--normalize', type=str, default='constant', help='normalization type (None "std", "constant", "kspace", or "max")')
    parser.add_argument('--std', type=float, default=0.02, help='standard dev of denoiser')
    parser.add_argument('--image-size', default=320, type=int, help='image size (this is bigger than 320x320)')
    parser.add_argument('--batch-size', default=32, type=int, help='Mini batch size') #Ted used 16
    parser.add_argument('--patch-size', default=320, type=int, help='training patch size')
    parser.add_argument('--val-patch-size', default=320, type=int, help='val patch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs') #old value 300
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate') # Ted's default was 1e-3
    parser.add_argument('--lr-step-size', type=int, default=20,
                        help='Period of learning rate decay') #old value 100
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    
    parser.add_argument('--num_of_top_slices', default=8, type=int, help='top slices have bigger brain image and less air region')
    
    parser.add_argument('--report-interval', type=int, default=6, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=int, default=0,
                        help='Which device to train on.')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='/home/bendel.8/Git_Repos/ComparisonStudy/utils/fastmri/models/PnP',
                        help='Path where model and results should be saved')
    # parser.add_argument('--data-path-train', type=pathlib.Path,
    #                     required=True)
    # parser.add_argument('--data-path-val', type=pathlib.Path,
    #                     required=True)
    parser.add_argument('--data-path', type=pathlib.Path,
                        required=True)
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument("--use-mid-slices", default=False, action='store_true', help="use only middle slices")
    parser.add_argument("--scanner-strength", type=float, default=None, help="Leave as None for all, >2.2 for 3, > 2.2 for 1.5")
    parser.add_argument("--scanner-mode", type=str, default=None, help="Leave as None for all, other options are PD, PDFS")
    parser.add_argument("--run-name", type=str, default=None, help="wandb save name")
    parser.add_argument("--rss-target", default=False, action='store_true', help="Use rss scaling")
    parser.add_argument('--mask-path', type=str, default=None, help='Path to mask (saved as Tensor)')
    parser.add_argument('--nc', type=int, default=4, help='number of coils to simulate')
    parser.add_argument('--coil-root', type=str, default='/home/reehorst.3/Documents/Reehorst_coil_maps/', help='path to coil directory')
    parser.add_argument("--train", default=True, action='store_true', help="to differentiate between training and val data; used while accessing the sens maps")
    return parser

if __name__ == '__main__':
    print("Running training code for the DnCNN denoiser for the Brain Data...")
    args = create_arg_parser().parse_args()
    # restrict visible cuda devices
    if args.data_parallel or (args.device >= 0):
        if not args.data_parallel:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" # change it accordingly: "0,1", "0,2,3", "0,1,2,3"
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    main(args)
