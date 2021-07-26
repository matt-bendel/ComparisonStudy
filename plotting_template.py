import matplotlib.pyplot as plt
import numpy as np

# Define number of rows and columns in subplot
rows = 2
cols = 3

def generate_image(fig, max, image, method, image_ind):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind) # Add to subplot
    ax.imshow(np.abs(image), cmap='gray') # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Assign x label for plot
    plt.xlabel(f'{method} Reconstruction')

def generate_error_map(fig, max, target, recon, method, image_ind, k=5):
    # Assume rows and cols are available globally
    # rows and cols are both previously defined ints
    ax = fig.add_subplot(rows, cols, image_ind) # Add to subplot

    # Normalize error between target and reconstruction
    error = np.abs(target - recon)
    normalized_error = error / error.max()

    im = ax.imshow(k * normalized_error, cmap='jet', vmin=0, vmax=1) # Plot image

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Assign x label for plot
    plt.xlabel(f'{method} Error')

    # Return plotted image and its axis in the subplot
    return im, ax

def get_colorbar(fig, im, ax):
    fig.subplots_adjust(right=0.85) # Make room for colorbar

    # Get position of final error map axis
    [[x10, y10], [x11, y11]] = ax.get_position().get_points()

    # Appropriately rescale final axis so that colorbar does not effect formatting
    pad = 0.01
    width = 0.02
    cbar_ax = fig.add_axes([x11 + pad, y10, width, y11 - y10])

    fig.colorbar(im, cax=cbar_ax) # Generate colorbar

# Assume general_recon and zfr_recon are bot previously defined 2D square numpy arrays (i.e. the reconstructed images)
# Assuming target is a previously defined 2D square numpy array (i.e. the ground truth image)
gt_max = target.max() # Get max value from ground truth
fig = plt.figure() # Create our figure
fig.suptitle('T2 Reconstructions')

# Generate the grayscale images of the GT and reconstructions
generate_image(fig, gt_max, target, 'GT', 1)
generate_image(fig, gt_max, zfr_recon,'ZFR', 2)
generate_image(fig, gt_max, general_recon, 'Some Method', 3)

# Generate error maps of reconstructions and grab the image and axis of the last error map for colorbar
generate_error_map(fig, gt_max, target, zfr_recon, 'ZFR', 5)
im, ax = generate_error_map(fig, gt_max, target, general_recon, 'Some Method', 6)

# Generate the colorbar
get_colorbar(fig, im, ax)

# Save the generated figure
plt.savefig('my/image/path.png')

