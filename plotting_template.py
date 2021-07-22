import matplotlib.pyplot as plt
import numpy as np

rows = 2
cols = 3

def generate_image(fig, max, image, method, image_ind):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    ax.imshow(np.abs(image), cmap='gray', extent=[0, max, 0, max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method} Reconstruction')

def generate_error_map(fig, max, target, recon, method, image_ind, k=5):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind)
    ax.imshow(k * np.abs(target - recon), cmap='jet', extent=[0, max, 0, max])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.xlabel(f'{method} Error')

# Assume general_recon and zfr_recon are bot previously defined numpy arrays
# Assuming target is a previously defined numpy array
gt_max = target.max()
fig = plt.figure()
fig.suptitle('T2 Reconstructions')
generate_image(fig, gt_max, target, 'GT', 1)
generate_image(fig, gt_max, zfr_recon,'ZFR', 2)
generate_image(fig, gt_max, general_recon, 'Some Method', 3)
generate_error_map(fig, gt_max, target, zfr_recon, 'ZFR', 5)
generate_error_map(fig, gt_max, target, general_recon, 'Some Method', 6)

