import os

import data_process
import datatr
import dataval
import datatest
import predict
import unet


# Path to the root folder in which all the different sets 
# have been saved. DO NOT ADD "/" AT THE END.
path = "D:/Users/imbrm/ISIC_2017-2""

# Make a copy of the datasets and save them as .npy
# They will be used later for the classification task.
os.makedirs("npy_datasets")
os.makedirs("npy_datasets/original")
datatr.to_npy(path, False)
dataval.to_npy(path, False)
datatest.to_npy(path, False)


# "D:/Users/imbrm/ISIC_2017-2"
# Augment, resize, greyscale the train and val sets.
# Resize and greyscale the test set.
# "jobs" is the number of parallel jobs to use:
# the more powerful your machine, the higher you can use.
jobs = 5
data_process.train_val_sets(path, jobs)


# Save the processed datasets as new .npy
os.makedirs("npy_datasets/processed")
datatr.to_npy(path, True)
dataval.to_npy(path, True)
datatest.to_npy(path, True)


# Create models for each segmentation network
size = (256,256)
unet = unet.unet(size, batch_norm=False)
unet_bn = unet.unet(size, batch_norm=True)
res_se_unet = unetpolished.get_unet()
focusnet = focusnetalpha.focusnet()

# Load weights for each model
unet.load_weights("")
unet_bn.load_weights("")
res_se_unet.load_weights("")
focusnet.load_weights("")

