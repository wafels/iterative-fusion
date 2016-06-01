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
import matplotlib.cm as cm
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


def get_psf(detector, function='test'):
    """
    Defines a PSF
    """
    if function == 'test':
        sigma = test_gaussian_width(detector)
        psf = test_psf(nx, sigma)
    else:
        psf = None

    return psf


def show_image_set(images, title=""):
    """
    Make a matplotlib plot of the image set
    :param images:
    :return: a plot of all the images
    """
    n_images = images.shape[0]
    ncols = int(np.ceil(np.sqrt(1.0 * n_images)))
    nrows = int(np.ceil(n_images/(1.0*ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    for nrow in range(0, nrows):
        for ncol in range(0, ncols):
            n = ncol + nrow*ncols
            if n <= n_images - 1:
                axes[nrow, ncol].imshow(images[n, :, :], cmap=cm.gray)
                axes[nrow, ncol].set_title('{:s} {:n}'.format(title, n+1))
    plt.show()


class DetectedImage:
    def __init__(self, detected, psf):
        self.detected = detected
        self.psf = psf


def ingaramo_deconvolution(measurement, psfs, n_iterations):

    n_images = measurement.shape[0]
    ny = measurement.shape[1]
    nx = measurement.shape[2]

    # Blurred estimate of the source image
    blurred_estimate = np.zeros_like(measurement)

    # The initial estimate
    estimate = np.ones((ny, nx))

    # Keep a history of the evolution of the estimate of the source
    source_estimate_history = np.zeros((n_iterations + 1, ny, nx))

    # Begin the deconvolution
    source_estimate_history[0, :, :] = estimate
    for i in range(n_iterations):
        print("Iteration", i)

        for j in range(n_images):
            # Take the estimated true image and apply the PSF
            psf = psfs[j, :, :]
            blurred_estimate[j, :, :] = fftconvolve(estimate, psf, 'same') # foxsi_convolve(this_detector, estimate, function=analysis_type) # np.convolve(psf[t], estimate)  # gaussian_filter(signal_range[t] * estimate, sigma=sigma_range[t])

        # Compute correction factor
        correction_factor = np.divide(measurement, blurred_estimate)

        print(" Blurring correction ratio...")
        for j in range(n_images):
            # Update the correction factor
            psf = psfs[j, :, :]
            correction_factor[j, :, :] = fftconvolve(correction_factor[j, :, :], psf, 'same')  # foxsi_convolve(this_detector, correction_factor[j, :, :], function=analysis_type)  # gaussian_filter(signal_range[t] * correction_factor[t, :, :], sigma=sigma_range[t])

        # Update the estimate
        estimate = np.multiply(estimate, correction_factor.mean(axis=0))

        # Save the evolution of the estimate
        print(source_estimate_history[i+1, :, :].min(), source_estimate_history[i, :, :].max())
        source_estimate_history[i+1, :, :] = estimate

    return source_estimate_history


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
        psf = test_psf(nx, gaussian_width)
        simulated_image = np.random.poisson(fftconvolve(test_image, psf, 'same'))
        observed_images[i, :, :] = simulated_image
        observed_image = DetectedImage(simulated_image, psf)

# Which images do we want to deconvolve
these_images = [0, 1, 2, 3, 4, 5, 6]
n_images = len(these_images)

# How many iterations
n_iterations = 10

# The measurements we are interested in
measurement = observed_images[these_images, :, :]

# PSFs
psfs = np.zeros((len(these_images), ny, nx))
for i, detector in enumerate(these_images):
    psfs[i, :, :] = get_psf(detector, function=analysis_type)

source_estimate_history = ingaramo_deconvolution(measurement, psfs, n_iterations)

show_image_set(measurement, title='Measurement #')
show_image_set(source_estimate_history[1:, :, :], title='Ingaramo i =')

# Sum all the images and their PSFs and deconvolve them
summed_measurement = np.zeros((1, ny, nx))
summed_measurement[0, :, :] = np.sum(measurement, axis=0).reshape(1, ny, nx)
summed_psf = np.zeros_like(summed_measurement)
summed_psf[0, :, :] = np.sum(psfs, axis=0).reshape(1, ny, nx)

source_estimate_history = ingaramo_deconvolution(summed_measurement, summed_psf, n_iterations)
show_image_set(source_estimate_history[1:, :, :], title='Summed i =')

# Do each image separately and add up the final results
final_images = np.zeros((n_images, ny, nx))
for i, detector in enumerate(these_images):
    this_measurement = np.zeros((1, ny, nx)).reshape(1, ny, nx)
    this_measurement[0, :, :] = measurement[detector, :, :].reshape(1, ny, nx)
    this_psf = np.zeros_like(this_measurement).reshape(1, ny, nx)
    this_psf[0, :, :] = psfs[detector, :, :].reshape(1, ny, nx)

    source_estimate_history = ingaramo_deconvolution(this_measurement,
                                                     this_psf,
                                                     n_iterations)

    final_images[detector, :, :] = source_estimate_history[-1, :, :]


final_image = np.sum(final_images, axis=0)
plt.figure()
plt.imshow(final_image, cmap=cm.gray)
plt.show()
