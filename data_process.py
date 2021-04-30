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



# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True


# Configure logging's details
logging.basicConfig(
    filename="data_process.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def del_superpixels(input_path, jobs):
    """Deletes the superpixels images of the skin lesions.

    Args:
        input_path (string): Path of the folder containing the superpixel images.
        jobs (string): Number of jobs to be used for parallelisation.
    """
    # Store the IDs of all the _superpixel images in a list.
    images = [
        splitext(file)[0]
        for file in listdir(input_path)
        if "_superpixels" in splitext(file)[0]
    ]
    print("Deleting Superpixel Images:")
    Parallel(n_jobs=jobs)(
        delayed(os.remove)(str(input_path + "/" + str(image + ".png")))
        for image in tqdm(images)
    )
    logging.info(f"Succesfully deleted {len(images)} SUPERPIXEL images.")


def resize(image, input_folder, size):
    """Defining a function that will allow me to parallelise the resizing process.
    It takes the name (basename) of the current image, resizes and saves the image.

    Args:
        input_path (string): Path to the image.
        size (tuple): Target size to be resized to.
    """
    image_path = input_folder + "/" + image + ".png"
    img = Image.open(image_path)
    img = img.resize((size[0], size[1]), resample=Image.ANTIALIAS)
    img.save(image_path)


def resize_set(input_folder, size, jobs, train_val, img_mask):
    """
    Stores the input and output directories, then stores all the
    names of the images in a list, and executes the resizing in parallel.
    For the parallelisation, Parallel and delayed are used.
    tqdm is used for visual representation of the progress, since the
    dataset is around 30GB, it will take some time to process.

    Args:
        input_folder (string): Path for input folder.
        size (tuple): Target size to be resized to.
        jobs (int): Number of parallelised jobs.
        train_val (string): Specifies whether it's "Train" or "Validation"; used for showing progress.
        is_mask (string): States whether it's an "Image" or "Mask" set; used for showing progress.
    """
    images = [splitext(file)[0] for file in listdir(input_folder)]
    print(f"Resizing {train_val} {img_mask}.")
    Parallel(n_jobs=jobs)(
        delayed(resize)(image, input_folder, size)
        for image in tqdm(images)
    )


def get_result(image_id, csv_file_path):
    """Checks whether the inputted image was a melanoma or not.

    Args:
        image_id (string): ID of the image.
        csv_file_path (string): Path leading to the .csv file with ground truth.

    Returns:
        melanoma (int): The melanoma classification result in 0 or 1.
    """
    df = pd.read_csv(csv_file_path)
    img_index = df.loc[df["image_id"] == image_id].index[0]
    melanoma = df.at[img_index, "melanoma"]
    return melanoma


def augment_operations(image_id, image_folder_path, mask_folder_path, train_val):
    """Performs augmentation operations on the inputted image.
    Seed is used for for applying the same augmentation to the image and its mask.

    Args:
        image_id (string): The ID of the image to be augmented.
        image_folder_path (string): Path of folder in which the augmented img will be saved.
        mask_folder_path (string): Path of folder in which the augmented mask will be saved.
        train_val (string): Specifies whether it's "Train" or "Validation".

    Returns:
        new_img (Image): New augmented PIL image.
        new_img_mask (Image): New augmented PIL mask.
    """
    mask_id = image_id + "_segmentation"
    img = Image.open(image_folder_path + "/" + image_id + ".png")
    mask = Image.open(mask_folder_path + "/" + mask_id + ".png")

    transf_comp = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=360, scale=(1, 1.7), shear=[0, 20, 0, 20], fillcolor=0
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomPerspective(p=1)
        ]
    )

    ##############################  <--- Comment out HERE to load seeds from .csv file 
    # Set random seed.
    seed = np.random.randint(0, 2**30)

    if train_val == "Validation":
        # Write seed in .csv file
        with open('seedval.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([image_id, seed])
    else:
        # Write seed in .csv file
        with open('seeds.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([image_id, seed])
    ##############################
    

    ############################## <--- Uncomment HERE to load seeds from .csv file 
    # # Load seed from file
    # if train_val == "Validation":
    #     df = pd.read_csv("seedval.csv")
    # else:
    #     df = pd.read_csv("seed.csv")

    # img_index = df.loc[df["image_id"] == image_id].index[0]
    # seed = df.at[img_index, "seed"]
    ##############################

    # Set Torch's seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Transform the image and mask using the same transformation.
    new_img = transf_comp(img)
    torch.manual_seed(seed)
    new_img_mask = transf_comp(mask)


    return new_img, new_img_mask


def augment_img(image_id, images_folder_path, masks_folder_path, csv_file_path, train_val):
    """Executes augmentation on a single image. Due to imbalanced dataset,
    it will perform more augmentation on melanoma images.
    If mole is not melanoma, perform 1 augmentation with probability=0.5.
    If mole is melanoma, perform 4 augmentation with probability=1.
    It performs the same transformation on the image and its relative mask.
    I chose a simple random number generator over PyTorch's RandomApply because
    this way an image that is not ment to be augmented will not be processed at all:
    when using RandomApply, the image will still be saved as ___x1 despite having
    recieved no augmentation i.e. being identical to the original picture.

    Args:
        image_id (string): ID of the image to be augmented.
        images_folder_path (string): Path of folder in which the augmented img will be saved.
        masks_folder_path (string): Path of folder in which the augmented mask will be saved.
        csv_file_path (string): Path leading to the .csv file with ground truth.
        train_val (string): Specifies whether it's "Train" or "Validation".
    """
    if train_val == "Validation":
        img_1, img_1_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path, train_val
        )

        # Save image and mask in two dedicated folders.
        img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
        img_1_mask.save(
            masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100)
        
        return
        
    else:
        melanoma = int(get_result(image_id, csv_file_path))
        if melanoma == 0:
            augm_probability = 0.5
            n = random.random()
            if n < augm_probability:
                # Perform augmentation, store the resulting image and mask.
                img_1, img_1_mask = augment_operations(
                    image_id, images_folder_path, masks_folder_path, train_val
                )

                # Save image and mask in two dedicated folders.
                img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
                img_1_mask.save(
                    masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100
                )

                # Add new datapoint to .csv file 
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([image_id + "x1", 0, 0])


        if melanoma == 1:
            # Perform augmentations, store the resulting images and masks.
            img_1, img_1_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path, train_val
            )
            img_2, img_2_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path, train_val
            )
            img_3, img_3_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path, train_val
            )
            img_4, img_4_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path, train_val
            )

            # Save images in dedicated folder.
            img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
            img_2.save(images_folder_path + "/" + image_id + "x2" + ".png", "PNG", quality=100)
            img_3.save(images_folder_path + "/" + image_id + "x3" + ".png", "PNG", quality=100)
            img_4.save(images_folder_path + "/" + image_id + "x4" + ".png", "PNG", quality=100)

            # Save masks in dedicated folder.
            img_1_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100
            )
            img_2_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x2" + ".png", "PNG", quality=100
            )
            img_3_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x3" + ".png", "PNG", quality=100
            )   
            img_4_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x4" + ".png", "PNG", quality=100
            )

            # Add new datapoint to .csv file 
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_id + "x1", 1, 0])
                writer.writerow([image_id + "x2", 1, 0])
                writer.writerow([image_id + "x3", 1, 0])
                writer.writerow([image_id + "x4", 1, 0])


def augment_dataset(images_folder_path, masks_folder_path, csv_file_path, jobs, train_val):
    """Performs augmentation on the whole dataset.
    Augmentation is performed in parallel to speed up process.

    Args:
        images_folder_path (string): Path to folder containing images of moles.
        masks_folder_path (string): Path to folder containing images of masks.
        csv_file_path (string): Path to .csv file containing ground truth.
        jobs (int): Number by which the parallelisation will be applied concurrently.
        train_val (string): Specifies whether it's "Train" or "Validation".
    """
    images = [splitext(file)[0] for file in listdir(images_folder_path)]
    print(f"Augmenting {train_val} Images and Masks:")
    Parallel(n_jobs=jobs)(
        delayed(augment_img)(
            image, images_folder_path, masks_folder_path, csv_file_path, train_val
        )
        for image in tqdm(images)
    )

    logging.info(f"Succesfully augmented {len(images)} images.")


def turn_grayscale(image, folder_path):
    """Function for parallelising the grayscale process.

    Args:
        image (string): ID of image to be turn into grayscale.
        folder_path (string): Path leading to folder containing images.
    """
    img = Image.open(folder_path + "/" + image + ".png")
    grey = transforms.functional.rgb_to_grayscale(img)
    grey.save(folder_path + "/" + image + ".png")


def make_greyscale(folder_path, jobs):
    """Turns all images in a folder from RGB to grayscale.

    Args:
        folder_path (string): Path leading to folder containing images.
        jobs (int): Number of job for parallelisation.
    """
    images = [splitext(file)[0] for file in listdir(folder_path)]
    print("Turning images to GrayScale:")
    Parallel(n_jobs=jobs)(
        delayed(turn_grayscale)(image, folder_path) for image in tqdm(images)
    )
    logging.info(f"Successfully turned {len(images)} images to GrayScale.")


def turn_np_imgs(folder_path):
    """Functions used for reading and returning the images in a list format.

    Args:
        folder_path (string): Path leading to folder containing images.

    Returns:
        imgs_array: (list): Multi-dimensional list containing the pixel values of images.
    """    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    for image in images:
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        im = im.tolist()
        imgs_array.append(im)
    #npa = np.asarray(imgs_array, dtype=np.float32)
    return imgs_array

def turn_np_masks(folder_path):
    """[summary]
    Functions used for reading and returning the masks in a list format.
    The pixel values are thresholded to either 0 or 1.

    Args:
        folder_path (string): Path leading to folder containing images.

    Returns:
        imgs_array (list): Multi-dimensional list containing the pixel values of masks.
        imgs_array1 (list): Multi-dimensional list containing the thresholded pixel values of masks. 
    """    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    imgs_array1 = []
    for image in images:
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        ret, im1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        ret2, im2 = cv2.threshold(im, 127, 1, cv2.THRESH_BINARY)
        im1 = im1.tolist()
        im2 = im2.tolist()
        imgs_array.append(im1)
        imgs_array1.append(im2)
    # npa = np.asarray(imgs_array, dtype=np.float32)
    # npa1 = np.asarray(imgs_array1, dtype=np.float32)
    
    #return npa, npa1
    return imgs_array, imgs_array1
    


def convert(image, folder):
    """Parallelisable function for converting all the images from JPEG to PNG format.

    Args:
        image (string): Image ID to convert.
        folder (string): Path of the folder containing images to convert.
    """    
    img = Image.open(folder + "/" + image + ".jpg")
    img.save(folder + "/" + image + ".png")
    os.remove(folder + "/" + image + ".jpg")


def convert_format(folder, jobs, train_or_val):
    """Converts all the images from JPEG to PNG format.

    Args:
        folder (string): Path of the folder containing images to convert.
        jobs (int): Number of jobs for parallelisation.
        train_or_val (string): Specifies whether it's Train or Validation images.
    """    
    images = [splitext(file)[0] for file in listdir(folder)]
    print(f"Converting {train_or_val} from JPEG to PNG.")
    Parallel(n_jobs=jobs)(
        delayed(convert)(image, folder)
        for image in tqdm(images)
    )


def generate_dataset(path, resize_dimensions, n_jobs):
    masks_suffix = "_GT_masks"
    csv_suffix = "_GT_result.csv"

    images_folder_path = path + "/" + "Train"
    masks_folder_path = images_folder_path + masks_suffix
    csv_file_path = images_folder_path + csv_suffix

    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    #convert_format(valimages_folder_path, 8, "Train")

    # Delete metadata file.
    try:
        os.remove(images_folder_path + "/" + "ISIC-2017_Training_Data_metadata.csv")
    except: 
        pass

    # Create new .csv file with seeds 
    with open('seeds.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "seed"])

    # Augment with relative masks.
    augment_dataset(
        images_folder_path,
        masks_folder_path,
        csv_file_path,
        n_jobs,
        "Train"
    )

    # Resize images.
    resize_set(
        images_folder_path,
        resize_dimensions,
        n_jobs,
        "Train",
        "Images",
    )

    # Resize masks.
    resize_set(
        masks_folder_path,
        resize_dimensions,
        n_jobs,
        "Train",
        "Masks",
    )

    # Make images greyscale.
    make_greyscale(
        images_folder_path,
        n_jobs,
    )

    ######################
    # VALIDATION 
        
    # Augment with relative masks.
    valimages_folder_path = path + "/" + "Validation"
    valmasks_folder_path = valimages_folder_path + masks_suffix

    # Delete superpixels.
    del_superpixels(valimages_folder_path, n_jobs)

    # Delete metadata file.
    try:
        os.remove(valimages_folder_path + "/" + "ISIC-2017_Validation_Data_metadata.csv")
    except: 
        pass

    # Create new .csv file with seeds for validation data
    with open('seedval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "seed"])

    #convert_format(valimages_folder_path, 8, "Validation")

    # Augment with relative masks.
    augment_dataset(
        valimages_folder_path,
        valmasks_folder_path,
        csv_file_path,
        n_jobs,
        "Validation"
    )

    # Resize images.
    resize_set(
        valimages_folder_path,
        resize_dimensions,
        n_jobs,
        "Validation",
        "Images"
    )

    # Resize masks.
    resize_set(
        valmasks_folder_path,
        resize_dimensions,
        n_jobs,
        "Validation",
        "Masks"
    )

    # Make images greyscale.
    make_greyscale(
        valimages_folder_path,
        n_jobs,
    )

    # print("Creating Validation folders:")
    # os.makedirs(path + "/" + "Validation", exist_ok=True)
    # os.makedirs(path + "/" + "Validation_GT_masks", exist_ok=True)

    # # Split
    # split_train_val(csv_file_path, 0.15)


def process_test_set(path, resize_dimensions, n_jobs):
    masks_suffix = "_GT_masks"
    csv_suffix = "_GT_result.csv"

    images_folder_path = path + "/" + "Test"
    masks_folder_path = images_folder_path + masks_suffix

    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    # Convert JPEG to PNG
    convert_format(valimages_folder_path, 8, "Train")

    # Delete metadata file.
    try:
        os.remove(images_folder_path + "/" + "ISIC-2017_Test_v2_Data_metadata.csv.csv")
    except: 
        pass

    # Resize images.
    resize_set(
        images_folder_path,
        resize_dimensions,
        n_jobs,
        "Test",
        "Images",
    )

    # Resize masks.
    resize_set(
        masks_folder_path,
        resize_dimensions,
        n_jobs,
        "Test",
        "Masks",
    )

    # Make images greyscale.
    make_greyscale(
        images_folder_path,
        n_jobs,
    )


def train_val_sets(path, jobs):
    size = (256,256)

    # del_augm(path + "/Train_GT_masks", 8)
    # del_augm(path + "/Validation_GT_masks", 8)
    generate_dataset(path, size, jobs)
    process_test_set(path, size, jobs)
