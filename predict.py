import os
import cv2

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from unet import unet

import tensorflow as tf
# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

path = "D:/Users/imbrm/ISIC_2017/check/new/all"
input_size = (256,256,1)
version = 0
weights_filename = 'unet_salt_weights_{}.h5'.format(version)

net = unet(input_size)
net.load_weights(weights_filename)

img = Image.open("D:/Users/imbrm/ISIC_2017/check/new/all/Validation/ISIC_0000034.png")
im = cv2.imread("D:/Users/imbrm/ISIC_2017/check/new/all/Validation/ISIC_0000034.png", 0)

print(im.shape)

im = im[np.newaxis,:,:,np.newaxis]

print(im.shape)

# plt.imshow(img)
# plt.waitforbuttonpress()
mask_test = net.predict(im)

plt.imshow(mask_test[0,:,:,0])
plt.waitforbuttonpress()