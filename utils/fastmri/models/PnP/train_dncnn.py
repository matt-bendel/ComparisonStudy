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

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from utils import fastmri
from utils.fastmri.data import transforms
from utils.fastmri.data.mri_data import SliceDataset
from dncnn import DnCNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, noise_std, mag_only=False, std_normalize=False, use_seed=True):
        # self.mask_func = mask_func
        # self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.mag_only = mag_only
        self.std_normalize = std_normalize
        self.noise_std = noise_std

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        # Inverse Fourier Transform to get image
        image = fastmri.ifft2c(kspace)

        # Absolute value
        if self.mag_only:
            image = fastmri.complex_abs(image).unsqueeze(2)

        image, scale = transforms.denoiser_normalize(image, is_complex=(not self.mag_only), use_std=self.std_normalize)

        # move real/imag to channel position
        image = image.permute(2, 0, 1)
        image = image.clamp(-1, 1)  # should only clamp if using std_normalization

        target = image
        # add zero mean noise
        image = image + self.noise_std * torch.randn(image.size())
        # target = target.clamp(-6, 6)
        return image, target, scale, attrs['norm'].astype(np.float32)


def create_datasets(args):
    train_data = SliceDataset(
        root=args.data_path / f'singlecoil_train',
        transform=DataTransform(args.std, mag_only=args.mag_only, std_normalize=args.std_normalize),
        sample_rate=1.0,
        challenge='singlecoil',
    )
    dev_data = SliceDataset(
        root=args.data_path / f'singlecoil_val',
        transform=DataTransform(args.std, mag_only=args.mag_only, std_normalize=args.std_normalize,
                                use_seed=True),
        sample_rate=1.0,
        challenge='singlecoil',
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=64,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=64,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=64,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, scale, norm = data
        input = input.to(args.device)
        target = target.to(args.device)

        output = model(input)
        loss = F.mse_loss(output, target, reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, scale, norm = data
            input = input.to(args.device)
            target = target.to(args.device)
            output = model(input)

            # norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            # loss = F.mse_loss(output / norm, target / norm, size_average=False)
            loss = F.mse_loss(output, target, reduction='sum')
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag, args):
        if not args.mag_only:
            image = fastmri.complex_abs(image.permute(0, 2, 3, 1)).unsqueeze(3).permute(0, 3, 1, 2)
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, scale, norm = data
            input = input.to(args.device)
            target = target.to(args.device)
            output = model(input)
            save_image(target, 'Target', args)
            save_image(output, 'Reconstruction', args)
            save_image(torch.abs(target - output), 'Error', args)
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
    if args.mag_only:
        chans = 1
    else:
        chans = 2
    model = DnCNN(
        image_channels=chans,
        n_channels=args.num_chans,
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
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
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
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

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--data_path",
        type=pathlib.Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument('--num-chans', type=int, default=64, help='Number of U-Net channels')
    parser.add_argument('--mag-only', action='store_true', help='denoise mag only')
    parser.add_argument('--std-normalize', action='store_true', help='normalize with std instead of maximum')
    parser.add_argument('--std', type=float, default=5.0 / 255, help='standard dev of denoiser')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
#    random.seed(args.seed)
#    np.random.seed(args.seed)
#    torch.manual_seed(args.seed)
    main(args)
