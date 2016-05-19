#
# Adapted from code.google.com/p/iterative-fusion
#


import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from astropy.io import readsav
plt.ion()


#
# Load in all the images
#
directory = os.path.expanduser('~/foxsi/data_2014/')
n_observed_images = 7
for i in range(0, 7):
    file_path = os.path.join(directory, 'map{:n}.sav'.format(i))
    this_image = readsav(file_path)
    data = this_image['map' + str(i)]
    ny = data.shape[0]
    nx = data.shape[1]
    if i == 0:
        observed_images = np.zeros((n_observed_images, ny, nx))

    observed_images[i, :, :] = data[:, :]


# Which images do we want to deconvolve
these_images = [0, 1]
n_images = len(these_images)

# How many iterations
n_iterations = 100

# The measurements we are interested in
measurement = observed_images[these_images, :, :]

# Blurred estimate of the source image
blurred_estimate = np.zeros_like(measurement)

# Keep a history of the evolution of the estimate of the source
source_estimate_history = np.zeros((n_iterations, ny, nx))

# The initial estimate
estimate = np.ones((ny, nx))

print("Deconvolving noisy object...")

for i in range(n_iterations):
    print(" Iteration", i)

    for j, this_detector in these_images:
        #
        # Take the estimated true image and apply the PSF
        #
        blurred_estimate[j, :, :] = foxsi_convolve(this_detector, estimate) # np.convolve(psf[t], estimate)  # gaussian_filter(signal_range[t] * estimate, sigma=sigma_range[t])
    #
    # Compute correction factor
    #
    correction_factor = np.divide(measurement, blurred_estimate)

    print(" Blurring correction ratio...")
    for j, this_detector in these_images:
        #
        # Update the correction factor
        #
        correction_factor[j, :, :] = foxsi_convolve(this_detector, correction_factor[j, :, :])  # gaussian_filter(signal_range[t] * correction_factor[t, :, :], sigma=sigma_range[t])

    estimate = np.multiply(estimate, correction_factor.mean(axis=0))

    # Save the evolution of the estimate
    source_estimate_history[i+1, :, :] = estimate


def foxsi_convolve(detector, image):
    """
    Convolves an input image with the PSF from a specified detector
    :param this_detector: the FOXSI detector we are interested in
    :param image: input image
    :return: the convolution of the input image with the PSF from the
    specified detector
    """
    pass