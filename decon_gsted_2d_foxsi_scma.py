#
# Adapted from code.google.com/p/iterative-fusion
#
# For use with FOXSI 2014 data
#


import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import readsav
plt.ion()


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def test_gaussian_width(i):
    return 100.0/(i+1)


def test_psf(nx, sigma):
    return gkern(nx, sigma)


def foxsi_convolve(detector, image, function='test'):
    """
    Convolves an input image with the PSF from a specified detector
    :param this_detector: the FOXSI detector we are interested in
    :param image: input image
    :return: the convolution of the input image with the PSF from the
    specified detector
    """
    if function == 'test':
        sigma = test_gaussian_width(detector)
        psf = test_psf(nx, sigma)
    else:
        pass

    answer = fftconvolve(image, psf, 'same')
    return answer


def transmogrify(img, sigma):
    """
    Make a noisy, blurred version of the input image
    :param img:
    :param sigma:
    :return:
    """
    nx = img.shape[0]
    psf = test_psf(nx, sigma)
    img2 = fftconvolve(img, psf, 'same')
    #img2 = gaussian_filter(sigma * img, sigma=sigma)
    return np.random.poisson(img2)
    return img2


def show_image_set(images, title):
    """
    Make a matplotlib plot of the image set
    :param images:
    :return: a plot of all the images
    """
    n_images = images.shape[0]
    ncols = int(np.ceil(np.sqrt(1.0 * n_images)))
    nrows = int(n_images/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for nrow in range(0, nrows):
        for ncol in range(0, ncols):
            n = ncol + nrow*ncols
            axes[nrow, ncol].imshow(images[n, :, :])
    plt.show()


analysis_type = 'test'

#
# Load in all the images
#
if not(analysis_type == 'test'):
    directory = os.path.expanduser('~/foxsi/foxsi-science/data_2014/')
    n_observed_images = 7
    for i in range(0, n_observed_images):
        file_path = os.path.join(directory, 'map{:n}.sav'.format(i))
        this_image = readsav(file_path)
        data = this_image['sv' + str(i)]
        ny = data.shape[0]
        nx = data.shape[1]
        if i == 0:
            observed_images = np.zeros((n_observed_images, ny, nx))

        observed_images[i, :, :] = data[:, :]
else:
    directory = os.path.expanduser('~/foxsi/foxsi-science/data_2014/')
    n_observed_images = 7
    test_image_filename = ''
    test_image = np.sum(mpimg.imread('/Users/ireland/foxsi/other/cameraman.jpg'),axis=2)
    ny = test_image.shape[0]
    nx = test_image.shape[1]

    for i in range(0, n_observed_images):
        if i == 0:
            observed_images = np.zeros((n_observed_images, ny, nx))

        gaussian_width = test_gaussian_width(i)
        observed_images[i, :, :] = transmogrify(test_image, gaussian_width)

# Which images do we want to deconvolve
these_images = [0, 1, 2, 3, 4, 5, 6]
n_images = len(these_images)

# How many iterations
n_iterations = 10

# The measurements we are interested in
measurement = observed_images[these_images, :, :]

# Blurred estimate of the source image
blurred_estimate = np.zeros_like(measurement)

# Keep a history of the evolution of the estimate of the source
source_estimate_history = np.zeros((n_iterations+1, ny, nx))

# The initial estimate
estimate = np.ones((ny, nx))

print("Ingaramo deconvolution of images of noisy source...")
source_estimate_history[0, :, :] = estimate
for i in range(n_iterations):
    print("Iteration", i)

    for j, this_detector in enumerate(these_images):
        #
        # Take the estimated true image and apply the PSF
        #
        blurred_estimate[j, :, :] = foxsi_convolve(this_detector, estimate, function=analysis_type) # np.convolve(psf[t], estimate)  # gaussian_filter(signal_range[t] * estimate, sigma=sigma_range[t])
    #
    # Compute correction factor
    #
    correction_factor = np.divide(measurement, blurred_estimate)

    print(" Blurring correction ratio...")
    for j, this_detector in enumerate(these_images):
        #
        # Update the correction factor
        #
        correction_factor[j, :, :] = foxsi_convolve(this_detector, correction_factor[j, :, :], function=analysis_type)  # gaussian_filter(signal_range[t] * correction_factor[t, :, :], sigma=sigma_range[t])

    estimate = np.multiply(estimate, correction_factor.mean(axis=0))

    # Save the evolution of the estimate
    print(source_estimate_history[i+1, :, :].min(), source_estimate_history[i, :, :].max())
    source_estimate_history[i+1, :, :] = estimate


# Deconvolve each image separately for comparison