import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


def to_npy(path, is_augmented):
    image_length = 224
    image_height = 224
    num_channels = 3
    i = 0

    if is_augmented:
        data_file = glob.glob(path + '/ISIC-2017_Test_v2_Data/*.png')
    else:
        data_file = glob.glob(path + '/ISIC-2017_Test_v2_Data/*.jpg')
    files = []

    data_file_mask = glob.glob(path + '/ISIC-2017_Test_v2_Part1_GroundTruth/*.png')

    trainData = np.zeros((len(data_file),image_length, image_height, num_channels))

    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    trainLabel = np.zeros((len(data_file_mask),image_length,image_height,1))


    for f in (data_file):
        a=cv2.imread(f)
        resized_image = cv2.resize(a, (image_length, image_height))
        resized_image = resized_image.astype(np.float64)
        trainData[i,:,:,:] = resized_image[:,:,:]
        base = os.path.basename(path + '/ISIC-2017_Test_v2_Data/" + f)
        fileName = os.path.splitext(base)[0]
        files.append(fileName)
        i += 1
        
    for k in (data_file_mask):
        base = os.path.basename(path + '/ISIC-2017_Test_v2_Part1_GroundTruth/' + k)
        fileName = os.path.splitext(base)[0]
        fileName = fileName[0:12] + fileName[25:]
        index = files.index(fileName)
        image = cv2.imread(k)
        gray = rgb2gray(image)
        resized_image = cv2.resize(gray, (224, 224))
        gray_image = img_to_array(resized_image)
        trainLabel[index, :, :, :] = gray_image[:, :, :]
        
    
    if is_augmented:
        folder = "/processed"
    else:
        folder = "/original"
    np.save(path + folder + '/dataval.npy',trainData)
    np.save(path + folder + '/dataMaskval.npy', trainLabel)