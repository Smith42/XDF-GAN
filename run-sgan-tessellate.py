"""
Script to create very large tessellated GDF

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
import imageio
from skimage.util import view_as_windows

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

    low_vals = np.random.permutation(ar[ar <= 2*sigma])
    ar[np.where(ar <= 2*sigma)] = low_vals

    return ar.astype(np.float32)


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Prorduce a fake xdf file.")
    # Args
    parser.add_argument("-m", "--model", help="Model file (h5).")
    parser.add_argument("-l", "--logdir", nargs="?", default="../big_ims", help="Logdir, default ../big_ims")
    parser.add_argument("-z", "--z_size", nargs="?", default=1024, type=int, help="Input noise array size (*16 for output size), default 1024. Must be a power of 2.")
    parser.add_argument("-o", "--overlap", nargs="?", default=32, type=int, help="Overlap between tiles in z space.")
    parser.add_argument("-f", "--fits", default=False, action="store_true", help="Output in FITS format.")
    parser.add_argument("-p", "--png", default=False, action="store_true", help="Output greyscale PNG images + histogram.")
    parser.add_argument("-n", "--numpy", default=False, action="store_true", help="Output numpy array.")
    args = parser.parse_args()

    dt = int(time())
    model_file = args.model
    logdir = "{}/{}/".format(args.logdir, dt)
    os.mkdir(logdir)
    z_size = args.z_size
    overlap = args.overlap
    chunks = z_size//64
    maxes = [0.5262004, 0.44799575, 0.62030375]
    mins = [-0.004748813, -0.0031752307, -0.011242471]

    # Load generator
    gen = load_model(model_file)

    big_z = np.random.randn(z_size+overlap, z_size+overlap, 50).astype(np.float32)
    mini_zs = np.squeeze(view_as_windows(big_z, ((z_size//chunks)+overlap, (z_size//chunks)+overlap, 50), step=(z_size//chunks, z_size//chunks, 1)))
    print(mini_zs.shape)
    z = np.reshape(mini_zs, (np.product(mini_zs.shape[0:2]), *mini_zs.shape[2:]))
    print(z.shape)

    print("Predicting imagery...")
    print("Batch size 4")
    ims = gen.predict(z, batch_size=4, verbose=1) # Batched for very large imagery
    print(logdir, ims.shape)

    ims = ims[:, (overlap*16)//2:-(overlap*16)//2, (overlap*16)//2:-(overlap*16)//2, :] # remove overlap
    ims = np.reshape(ims, (*mini_zs.shape[0:2], 1024, 1024, 3))

    im = np.concatenate(np.split(ims, len(ims), axis=0), axis=2) # Stitch image back together
    im = np.squeeze(np.concatenate(np.split(im, len(ims), axis=1), axis=3)) # ditto...

    # Output values
    if args.numpy: # Output n-channel image in npy format
        print("Outputting as npy")
        np.save("{}array.npy".format(logdir), np.squeeze(im))

    if args.png: # Output PNG images for each channel + a histogram for each (n-channel) image
        print("Outputting as PNG")
        hist = np.histogram(im, 10000)
        plt.yscale("log")
        plt.plot(hist[1][:-1], hist[0])
        plt.savefig("{}hist.png".format(logdir))
        plt.close()
        for channel in np.arange(ims.shape[-1]):
            plt.figure(figsize=(32, 32))
            plt.tight_layout()
            plt.imshow(np.squeeze(im[..., channel]))
            plt.savefig("{}{}.png".format(logdir, channel))
            plt.close()

    if args.fits: # Output as a separate FITS image for each channel
        print("Outputting as FITS")
        #im = un_min_max_norm(im, ar_max=0.4142234, ar_min=-0.011242471) # Uncomment for image wise norming
        for channel in np.arange(ims.shape[-1]):
             print("Channel:", channel)
             print("Before unnorming:", im[..., channel].max(), im[..., channel].min())
             im[..., channel] = un_min_max_norm(im[..., channel], ar_max=maxes[channel], ar_min=mins[channel]) # For channel wise norming
             im[..., channel] = rescale(im[..., channel])
             print("After unnorming:", im[..., channel].max(), im[..., channel].min())
             #pyfits.writeto("{}{}.fits".format(logdir, channel), np.squeeze(shuffle_noise_given_array(im[..., channel])), overwrite=True)
             pyfits.writeto("{}{}.fits".format(logdir, channel), np.squeeze(im[..., channel]), overwrite=True)

