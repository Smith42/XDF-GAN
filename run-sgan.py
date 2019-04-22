"""
Script to run GDF generation

Copyright 2019 Mike Smith
Please see COPYING for licence details
"""
import matplotlib as mpl
mpl.use("Agg")

# General imports
import numpy as np
import h5py
import os
from time import time
import argparse
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ML specific imports
from keras.models import load_model

def un_min_max_norm(ar, ar_max, ar_min):
    """
    Reverse min max normalising carried out on the original UDF data.
    """
    return ar*(ar_max - ar_min) + ar_min

def find_nearest(ar, val):
    """
    Get position in array of value nearest to 'val'.
    """
    return np.argmin(np.abs(ar - val))

def get_sigma(hwhm):
    """
    Given the half width at half maximum, find the standard deviation of a normal distribution.
    """
    return (2*np.abs(hwhm))/(np.sqrt(8*np.log(2)))

def noise_replacement_low_vals(x, sigma, mu):
    """
    Replace low values with a random normal distribution
    """
    return np.random.normal(mu, sigma) if np.abs(x) <= 2*sigma else x

def apply_noise_low_vals(ar):
    """
    Apply noise to low values given an array.
    """
    hist = np.histogram(ar, 100000)
    maxpoint = np.max(hist[0])
    negsx = hist[1][:-1][hist[1][:-1] <= 0]
    negsy = hist[0][hist[1][:-1] <= 0]

    hwhm = negsx[find_nearest(negsy, maxpoint/2)]
    sigma = get_sigma(hwhm)
    mu = 0

    ar_replaced_noise = noise_replacement_low_vals(ar, sigma, mu)
    return ar_replaced_noise.astype(np.float32)

def noise_replacement_all_vals(x, sigma, mu):
    """
    Add a noise sampled from a gaussian to all values
    """
    return x + np.random.normal(mu, sigma)


def apply_noise_all_vals(ar):
    """
    Apply additive noise to all values given an array.
    """
    hist = np.histogram(ar, 100000)
    maxpoint = np.max(hist[0])
    negsx = hist[1][:-1][hist[1][:-1] <= 0]
    negsy = hist[0][hist[1][:-1] <= 0]

    hwhm = negsx[find_nearest(negsy, maxpoint/2)]
    sigma = get_sigma(hwhm)
    mu = 0

    ar_replaced_noise = noise_replacement_all_vals(ar, sigma, mu)
    return ar_replaced_noise.astype(np.float32)

def rescale(ar):
    """
    Rescale so peak is at zero.
    """
    hist = np.histogram(ar, 10000)
    delta = hist[1][hist[0].argmax()]
    return ar - delta

def shuffle_noise_given_array(ar):
    """
    Shuffle noise values given an array.
    """
    hist = np.histogram(ar, 100000)
    maxpoint = np.max(hist[0])
    negsx = hist[1][:-1][hist[1][:-1] <= 0]
    negsy = hist[0][hist[1][:-1] <= 0]

    hwhm = negsx[find_nearest(negsy, maxpoint/2)]
    sigma = get_sigma(hwhm)
    mu = 0

    low_vals = np.random.permutation(ar[ar <= 1*sigma])
    ar[np.where(ar <= 1*sigma)] = low_vals

    return ar.astype(np.float32)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Produce a fake xdf file.")
    # Args
    parser.add_argument("-m", "--model", help="Model file (h5).")
    parser.add_argument("-l", "--logdir", nargs="?", default="../logs/outs", help="Logdir, default ../logs/outs/$UNIXTIME")
    parser.add_argument("-z", "--z_size", nargs="?", default=64, type=int, help="Input noise array size (*16 for output size), default 64.")
    parser.add_argument("-n", "--images", nargs="?", default=10, type=int, help="Number of images to generate.")
    parser.add_argument("-f", "--fits", default=False, action="store_true", help="Output in FITS format.")
    parser.add_argument("-p", "--png", default=False, action="store_true", help="Output greyscale PNG images + histogram.")
    parser.add_argument("--numpy", default=False, action="store_true", help="Output numpy array.")
    parser.add_argument("-s", "--shuffle", default=False, action="store_true", help="Shuffle output to mitigate noise waffling in FITS output.")
    parser.add_argument("--seed", nargs="?", default=42, type=int, help="A seed for np.random.seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    dt = int(time())
    model_file = args.model
    n_images = args.images
    logdir = "{}/{}/".format(args.logdir, dt)
    os.mkdir(logdir)
    z_size = args.z_size
    test_batch_size = 100
    # These are the original image maxima and minima for each channel
    maxes = [0.5262004, 0.44799575, 0.62030375]
    mins = [-0.004748813, -0.0031752307, -0.011242471]
    noise_replacement_low_vals = np.vectorize(noise_replacement_low_vals)
    noise_replacement_all_vals = np.vectorize(noise_replacement_all_vals)

    # Load generator
    gen = load_model(model_file)

    z = np.random.randn(n_images, z_size, z_size, 50).astype(np.float32)
    ims = gen.predict(z, batch_size=1, verbose=1) # added dtype still needs testing
    print(logdir, ims.shape, ims.dtype)

    # Output values
    for i, im in enumerate(ims):
        if args.numpy: # Output n-channel image in npy format
            print("Outputting as npy")
            np.save("{}{}.npy".format(logdir, i), np.squeeze(im))

        if args.png: # Output PNG images for each channel + a histogram for each (n-channel) image
            print("Outputting as PNG")
            hist = np.histogram(im, 10000)
            plt.yscale("log")
            plt.plot(hist[1][:-1], hist[0])
            plt.savefig("{}{}-hist.png".format(logdir, i))
            plt.close()
            for channel in np.arange(ims.shape[-1]):
                plt.figure(figsize=(16, 16))
                plt.imshow(np.squeeze(im[..., channel]), norm=LogNorm())
                plt.savefig("{}{}-{}.png".format(logdir, i, channel))
                plt.close()

        if args.fits: # Output as a separate FITS image for each channel
            print("Outputting as FITS")
            #im = un_min_max_norm(im, ar_max=0.4142234, ar_min=-0.011242471) # Uncomment for image wide norming
            for channel in np.arange(ims.shape[-1]):
                print("Channel:", channel)

                print("Before unnorming:", im[..., channel].max(), im[..., channel].min())
                im[..., channel] = un_min_max_norm(im[..., channel], ar_max=maxes[channel], ar_min=mins[channel]) # For channel wise norming
                im[..., channel] = rescale(im[..., channel])
                print("After unnorming:", im[..., channel].max(), im[..., channel].min())

                if args.shuffle:
                    pyfits.writeto("{}{}-{}.fits".format(logdir, i, channel), np.squeeze(shuffle_noise_given_array(im[..., channel])), overwrite=True)
                else:
                    pyfits.writeto("{}{}-{}.fits".format(logdir, i, channel), np.squeeze(im[..., channel]), overwrite=True)
