import os
from os.path import splitext
from os import listdir
import glob
import logging
import random
import shutil
import math
import cv2
import csv

import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

from torchvision import transforms
import torch
import matplotlib.pyplot as plt

images_folder_path = "D:/Users/imbrm/ISIC_2017-2/hehesmall/Train"
masks_folder_path = "D:/Users/imbrm/ISIC_2017-2/hehesmall/Train_GT_masks"
save_folder_path = "D:/Users/imbrm/ISIC_2017-2/hehesmall/end"

images = [splitext(file)[0] for file in listdir(images_folder_path)]
masks = [splitext(file)[0] for file in listdir(masks_folder_path)]
masks_array = []

for image, mask in zip(images,masks):
    image_path = images_folder_path + "/" + image + ".png"
    mask_path = masks_folder_path + "/" + mask + ".png"

    # image_path = "D:/Users/imbrm/ISIC_2017-2/hehesmall/Train/ISIC_0000019.png"
    # mask_path = "D:/Users/imbrm/ISIC_2017-2/hehesmall/Train_GT_masks/ISIC_0000019_segmentation.png"
    

    img = Image.open(image_path)
    mask = Image.open(mask_path)

    image_ = np.array(img)
    image_ = image_.astype(np.float64)
    mask_ = np.array(mask)
    mask_ = mask_.astype(np.float64)

    ret2, mask1_ = cv2.threshold(mask_, 127, 1, cv2.THRESH_BINARY)

    # plt.imshow(mask_)
    # plt.waitforbuttonpress()
    # plt.imshow(mask1_)
    # plt.waitforbuttonpress()

    assert image_.shape == mask1_.shape

    newImage = image_ * mask1_
    newImage = Image.fromarray(newImage)
    newImage = newImage.convert("L")
    newImage.save(save_folder_path + "/" + image + ".png")
