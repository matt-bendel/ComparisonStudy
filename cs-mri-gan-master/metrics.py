from skimage import measure
import matplotlib.pyplot as plt
import numpy as np

def metrics(true_tensor, test_tensor,max_val):
    psnrt = 0
    ssimt = 0
    plot = False
    for i in range(true_tensor.shape[0]): 
        psnr = measure.compare_psnr(true_tensor[i,:,:], test_tensor[i,:,:],data_range=max_val)
        if psnr < 26 and not plot:
            plot = True
            plt.figure()
            plt.imshow(np.abs(test_tensor[i,:,:]), cmap='gray')
            plt.savefig('31db_recon.png')
            plt.figure()
            plt.imshow(np.abs(true_tensor[i,:,:]), cmap='gray')
            plt.savefig('31db_gt.png')

        ssim = measure.compare_ssim(true_tensor[i,:,:], test_tensor[i,:,:],data_range=max_val)
        psnrt = psnrt+psnr
        ssimt = ssimt+ssim

    psnrt = psnrt/true_tensor.shape[0]
    ssimt = ssimt/true_tensor.shape[0]
    return psnrt, ssimt
