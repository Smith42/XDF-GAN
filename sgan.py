"""
Script to train GDF-SGAN

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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

# ML specific imports
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, LeakyReLU, ELU, GlobalAveragePooling2D, Concatenate
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator

def get_images(file):
    """
    Get XDF fits (np) file.
    """
    im = np.load(file)
    print(im.shape)
    return im

def random_crop(img, crop_size=128):
    """
    Random crop big xdf image.
    """
    height, width = img.shape[0], img.shape[1]
    x = np.random.randint(0, width - crop_size + 1)
    y = np.random.randint(0, height - crop_size + 1)
    return img[y:(y+crop_size), x:(x+crop_size), :]

def gen(z_shape=(None, None, 50), num_layers=4):
    """
    Model a spatial GAN generator with `num_layers` hidden layers.
    """
    fs = [32*2**f for f in np.arange(num_layers)][::-1] # define filter sizes

    z = Input(shape=z_shape) # z
    ct = Conv2DTranspose(filters=fs[0], kernel_size=4, strides=2, padding="same")(z)
    ct = ELU()(ct)

    for f in fs[1:]:
        ct = Conv2DTranspose(filters=f, kernel_size=4, strides=2, padding="same")(ct)
        ct = ELU()(ct)

        ct = Conv2DTranspose(filters=f, kernel_size=4, strides=1, padding="same")(ct)
        ct = ELU()(ct)

        ct = Conv2DTranspose(filters=f, kernel_size=4, strides=1, padding="same")(ct)
        ct = ELU()(ct)

    G_z = Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding="same", activation="sigmoid")(ct)
    model = Model(z, G_z, name="Generator")
    model.summary()
    return model

def disc(x_shape=(None, None, 6), num_layers=4):
    """
    Model a spatial GAN discriminator.
    """
    fs = [32*2**f for f in np.arange(num_layers)] # define filter sizes

    x = Input(shape=x_shape)

    c = Conv2D(filters=fs[0], kernel_size=4, strides=2, padding="same")(x)
    c = LeakyReLU(0.1)(c)

    for f in fs[1:]:
        c = Conv2D(filters=f, kernel_size=4, strides=2, padding="same")(c)
        c = LeakyReLU(0.1)(c)

    gap = GlobalAveragePooling2D()(c)
    y = Dense(1)(gap)
    model = Model(x, y, name="Discriminator")
    model.summary()
    return model

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser("Run a spatial GAN on XDF FITS data.")
    # Args
    parser.add_argument("-f", "--im_file", nargs="?", default="./data/mc_channelwise_clipping.npy", help="Numpy file containing image data.")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size, default 32.")
    parser.add_argument("-e", "--epochs", type=int, default=10001, help="Number of training epochs, default 301.")
    parser.add_argument("-l", "--logdir", nargs="?", default="./logs", help="Logdir, default ./logs")
    parser.add_argument("-r", "--learning_rate", nargs="?", type=float, default=0.0002, help="Learning rate for ADAM op")
    parser.add_argument("-d", "--debug", dest="debug", default=False, action="store_true", help="Print example images/histograms at every epoch")
    parser.add_argument("--gen_weights", nargs="?", help="File containing gen weights for continuation of training.")
    parser.add_argument("--disc_weights", nargs="?", help="File containing disc weights for continuation of training.")
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    debug = args.debug
    disc_weights = args.disc_weights
    gen_weights = args.gen_weights

    dt = int(time())
    logdir = "{}/{}/".format(args.logdir, dt)
    print("logdir:", logdir)
    os.mkdir(logdir)
    sizes = [(4, 64), (8, 128), (16, 256)] # Possible input and output sizes
    test_batch_size = (1, 32, 32, 50)

    # might want to alter learning rate...
    adam_op = Adam(lr=args.learning_rate, beta_1=0.5, beta_2=0.999)
    xdf = get_images(args.im_file)[..., 1:4] # take F606W, F775W and F814W channels
    og_histo = np.histogram(xdf, 10000)

    # Define generator and discriminator models
    gen = gen()
    disc = disc()

    if disc_weights is not None and gen_weights is not None:
        gen.load_weights(gen_weights)
        disc.load_weights(disc_weights)

    # Define real and fake images
    raw_reals = Input(shape=(None, None, 3))
    reals = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=0))(raw_reals)
    reals = Concatenate(axis=-1)([reals[0], reals[1]])
    z = Input(shape=(None, None, 50))
    fakes = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=0))(gen(z))
    fakes = Concatenate(axis=-1)([fakes[0], fakes[1]])
    disc_r = disc(reals) # C(x_r)
    disc_f = disc(fakes) # C(x_f)

    # Define generator and discriminator losses according to RaGAN described in Jolicoeur-Martineau (2018).
    # Dummy predictions and trues are needed in Keras (see https://github.com/Smith42/keras-relativistic-gan).
    def rel_disc_loss(y_true, y_pred):
        epsilon = 1e-9
        return K.abs(-(K.mean(K.log(K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)\
                 +K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)))

    def rel_gen_loss(y_true, y_pred):
        epsilon = 1e-9
        return K.abs(-(K.mean(K.log(K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)\
                 +K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)))

    # Define trainable generator and discriminator
    gen_train = Model([z, raw_reals], [disc_r, disc_f])
    disc.trainable = False
    gen_train.compile(adam_op, loss=[rel_gen_loss, None])
    gen_train.summary()

    disc_train = Model([z, raw_reals], [disc_r, disc_f])
    gen.trainable = False
    disc.trainable = True
    disc_train.compile(adam_op, loss=[rel_disc_loss, None])
    disc_train.summary()

    # Train RaGAN
    gen_loss = []
    disc_loss = []

    dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

    test_z = np.random.randn(test_batch_size[0],\
                             test_batch_size[1],\
                             test_batch_size[2],\
                             test_batch_size[3]).astype(np.float32)

    # Define batch flow
    batchflow = ImageDataGenerator(rotation_range=0,\
                                   width_shift_range=0.0,\
                                   height_shift_range=0.0,\
                                   shear_range=0.0,\
                                   zoom_range=0.0,\
                                   channel_shift_range=0.0,\
                                   fill_mode='reflect',\
                                   horizontal_flip=True,\
                                   vertical_flip=True,\
                                   rescale=None)

    start_time = time()
    for epoch in np.arange(epochs):
        print(epoch, "/", epochs)
        n_batches = 30 # int(len(ims) // batch_size)

        prog_bar = Progbar(target=n_batches)
        batch_start_time = time()

        for index in np.arange(n_batches):
            size = sizes[np.random.randint(len(sizes))]
            prog_bar.update(index)

            # Update G
            image_batch = batchflow.flow(np.array([random_crop(xdf, size[1]) for i in np.arange(batch_size)]), batch_size=batch_size)[0]
            z = np.random.randn(batch_size, size[0], size[0], 50).astype(np.float32)
            disc.trainable = False
            gen.trainable = True
            gen_loss.append(gen_train.train_on_batch([z, image_batch], dummy_y))

            # Update D
            image_batch = batchflow.flow(np.array([random_crop(xdf, size[1]) for i in np.arange(batch_size)]), batch_size=batch_size)[0]
            z = np.random.randn(batch_size, size[0], size[0], 50).astype(np.float32)
            disc.trainable = True
            gen.trainable = False
            disc_loss.append(disc_train.train_on_batch([z, image_batch], dummy_y))

        print("\nEpoch time", int(time() - batch_start_time))
        print("Total elapsed time", int(time() - start_time))
        print("Gen, Disc losses", gen_loss[-1], disc_loss[-1])

        ## Print out losses and pics of G(z) outputs ##
        if debug or epoch % 5 == 0:
            gen_image = gen.predict(test_z)
            print("OG im: max, min, mean, std", xdf.max(), xdf.min(), xdf.mean(), xdf.std())
            print("Gen im: max, min, mean, std", gen_image.max(), gen_image.min(), gen_image.mean(), gen_image.std())
            # Plot generated/real histo comparison
            gen_histo = np.histogram(gen_image, 10000)
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16, 16))
            axs.set_yscale("log")
            axs.plot(og_histo[1][:-1], og_histo[0], label="Original")
            axs.plot(gen_histo[1][:-1], gen_histo[0], label="Generated")
            axs.legend()
            plt.savefig("{}/{:05d}-histogram.png".format(logdir, epoch))
            plt.close(fig)

            # Plot generated image
            fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
            axs[0, 0].imshow(gen_image[0, ..., 0], cmap="gray", norm=LogNorm())
            axs[0, 1].imshow(gen_image[0, ..., 1], cmap="gray", norm=LogNorm())
            axs[0, 2].imshow(gen_image[0, ..., 2], cmap="gray", norm=LogNorm())
            #axs[1, 0].imshow(gen_image[0, ..., 3], cmap="gray", norm=LogNorm())
            #axs[1, 1].imshow(gen_image[0, ..., 4], cmap="gray", norm=LogNorm())
            axs[1, 0].imshow(gen_image[0], norm=LogNorm()) # was [1,2] and sliced [...,1:4]
            plt.tight_layout()
            plt.savefig("{}/{:05d}-example.png".format(logdir, epoch))
            plt.close(fig)

        ## Save model ##
        if epoch % 10 == 0:
            gen.save("{}/{:05d}-gen-model.h5".format(logdir, epoch))
            gen.save_weights("{}/{:05d}-gen-weights.h5".format(logdir, epoch))
            disc.save_weights("{}/{:05d}-disc-weights.h5".format(logdir, epoch))

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        disc_loss_ar = np.array(disc_loss)[:, 0]
        gen_loss_ar = np.array(gen_loss)[:, 0]
        axs.set_title("Losses at epoch " + str(epoch))
        axs.set_xlabel("Global step")
        axs.set_ylabel("Loss")
        axs.set_yscale("log")
        axs.plot(disc_loss_ar, label="disc loss")
        axs.plot(gen_loss_ar, label="gen loss")
        axs.legend()
        plt.savefig("{}/{:05d}-loss.png".format(logdir, epoch))
        plt.close(fig)
