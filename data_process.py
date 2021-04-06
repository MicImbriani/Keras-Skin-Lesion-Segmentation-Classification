import os
from os.path import splitext
from os import listdir
import glob
import logging
import random
import shutil
import math
import cv2

import pandas as pd
from sklearn import model_selection
from torch.utils.data import Dataset
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

from torchvision import transforms
import torch


# from dataset import MyDataset


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
        input_path (string): Path of the folder containing all the images.
        jobs (string): Number of job for parallelisation.
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


def resize_set(input_folder, size, jobs, train_or_test):
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
        train_or_test (string): States whether it's train or test set.
    """
    images = [splitext(file)[0] for file in listdir(input_folder)]
    print(f"Resizing {train_or_test} Images:")
    Parallel(n_jobs=jobs)(
        delayed(resize)(image, input_folder, size)
        for image in tqdm(images)
    )
    logging.info(f"Resized {len(input_folder)} {train_or_test} images.")


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


def augment_operations(image_id, image_folder_path, mask_folder_path):
    """Performs augmentation operations on the inputted image.
    Seed is used for for applying the same augmentation to the image and its mask.

    Args:
        image_id (string): The ID of the image to be augmented.
        image_folder_path (string): Path of folder in which the augmented img will be saved.
        mask_folder_path (string): Path of folder in which the augmented mask will be saved.

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
                degrees=360, scale=(0.5, 1.5), shear=[0, 20, 0, 20], fillcolor=0
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomPerspective(p=1)
        ]
    )

    # Set random seed.
    seed = np.random.randint(0, 2**30)
    random.seed(seed)
    torch.manual_seed(seed)

    # Transform the image and mask using the same transformation.
    new_img = transf_comp(img)
    torch.manual_seed(seed)
    new_img_mask = transf_comp(mask)


    return new_img, new_img_mask


def augment_img(image_id, images_folder_path, masks_folder_path, csv_file_path):
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
    """
    melanoma = int(get_result(image_id, csv_file_path))
    img = Image.open(images_folder_path + "/" + image_id + ".jpg")
    img.save(images_folder_path + "/" + image_id + ".png")
    os.remove(images_folder_path + "/" + image_id + ".jpg")
    if melanoma == 0:
        augm_probability = 0.5
        n = random.random()
        if n < augm_probability:
            # Perform augmentation, store the resulting image and mask.
            img_1, img_1_mask = augment_operations(
                image_id, images_folder_path, masks_folder_path
            )

            # Save image and mask in two dedicated folders.
            img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
            img_1_mask.save(
                masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100
            )

    if melanoma == 1:
        # Perform augmentations, store the resulting images and masks.
        img_1, img_1_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_2, img_2_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_3, img_3_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
        )
        img_4, img_4_mask = augment_operations(
            image_id, images_folder_path, masks_folder_path
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


def augment_dataset(images_folder_path, masks_folder_path, csv_file_path, jobs):
    """Performs augmentation on the whole dataset.
    Augmentation is performed in parallel to speed up process.

    Args:
        images_folder_path (string): Path to folder containing images of moles.
        masks_folder_path (string): Path to folder containing images of masks.
        csv_file_path (string): Path to .csv file containing ground truth.
        jobs (int): Number by which the parallelisation will be applied concurrently.
    """
    images = [splitext(file)[0] for file in listdir(images_folder_path)]
    Parallel(n_jobs=jobs)(
        delayed(augment_img)(
            image, images_folder_path, masks_folder_path, csv_file_path
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


def turn_np(folder_path, resize_dimensions):
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    for image in images:
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        im = im.tolist()
        imgs_array.append(im)
    npa = np.asarray(imgs_array, dtype=np.float32)
    return npa


def move_data(list, path, data_type):
    """Move the images whose ID is in "list" from the Training folder to Validation,
    or, in the case of masks, from Training_GT_masks to Validation_GT_masks.

    Args:
        list (list): List containing the IDs of validation images/masks.
        path (string): Path to parent folder.
        data_type (string): Defines whether it's an image or a mask.
    """
    input_folder = path + "/" + "Train"
    output_folder = path + "/" + "Validation"

    if data_type.capitalize() == "Image":
        for image_id in list:
            shutil.move(
                input_folder + "/" + image_id + ".png",
                output_folder + "/" + image_id + ".png",
            )

    if data_type.capitalize() == "Mask":
        input_folder = input_folder + "_GT_masks"
        output_folder = output_folder + "_GT_masks"
        for image_id in list:
            shutil.move(
                input_folder + "/" + image_id + ".png",
                output_folder +  "/" + image_id + ".png"
            )


def split(df, result, val_ratio, csv_file_path):
    """Performs the split into train and validation data.
    Stores the indices of images with or without melanoma in the "train" list,
    then it moves a certain percentage of them (specified by val_ratio) into
    the "validation" list. Finally, marks each image with "T" or "V" appropriately.
    The split is randomly performed using random.sample() function.

    Args:
        df (DataFrame): Pandas DataFrame containing information about the dataset.
        result (int): Whether it's melanoma (1) or no melanoma (0).
        val_ratio (float): Percentage of data to be split into validation.
        csv_file_path (string): File path for extrapolating parent path.

    Returns:
        df (DataFrame): The transformed DataFrame with marked images.
    """
    train = list(df[df["melanoma"] == result].index)
    # ceil function for rounding up float numbers
    n_val = math.ceil(val_ratio * len(train))
    validation = random.sample(train, n_val)
    validation.sort()
    for element in validation:
        train.pop(train.index(element))

    # Mark validation images with "V"
    for id in tqdm(validation):
        df.at[id, "split"] = "V"
    # Mark train images with "T"
    for id in tqdm(train):
        df.at[id, "split"] = "T"
    val_ids = [df.at[index, "image_id"] for index in validation]
    val_masks_ids = [df.at[index, "image_id"] + "_segmentation" for index in validation]

    path = os.path.split(csv_file_path)[0]

    # Move validation images to folder.
    move_data(val_ids, path, "Image")

    # Move validation masks to folder.
    move_data(val_masks_ids, path, "Mask")

    return df


def split_train_val(csv_file_path, percent):
    """Callable function for splitting the dataset into train and validation.

    Args:
        csv_file_path (string): Path to .csv file containing ground truth.
    """
    csv_name = splitext(os.path.basename(csv_file_path))[0]
    csv_copy_path = os.path.split(csv_file_path)[0] + "/" + csv_name + "_split.csv"
    shutil.copy2(csv_file_path, csv_copy_path)

    csv_copy = pd.read_csv(csv_copy_path)
    csv_copy["split"] = ""

    print("Splitting the dataset into Train/Validation:")

    # MELANOMA YES (result=1)
    csv_copy = split(csv_copy, 1, percent, csv_file_path)
    # MELANOMA NO (result=0)
    csv_copy = split(csv_copy, 0, percent, csv_file_path)

    csv_copy.to_csv(csv_copy_path, index=False)


def generate_dataset(path, resize_dimensions, n_jobs):
    masks_suffix = "_GT_masks"
    csv_suffix = "_GT_result.csv"

    images_folder_path = path + "/" + "Train"
    masks_folder_path = images_folder_path + masks_suffix
    csv_file_path = images_folder_path + csv_suffix

    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    # Delete metadata file.
    os.remove(images_folder_path + "/" + "ISIC-2017_Training_Data_metadata.csv")

    # Augment with relative masks.
    augment_dataset(
        images_folder_path,
        masks_folder_path,
        csv_file_path,
        n_jobs,
    )

    # Resize images.
    resize_set(
        images_folder_path,
        resize_dimensions,
        n_jobs,
        "Train",
    )

    # Resize masks.
    resize_set(
        masks_folder_path,
        resize_dimensions,
        n_jobs,
        "Masks",
    )

    # Make images greyscale.
    make_greyscale(
        images_folder_path,
        n_jobs,
    )

    print("Creating Validation folders:")
    os.makedirs(path + "/" + "Validation", exist_ok=True)
    os.makedirs(path + "/" + "Validation_GT_masks", exist_ok=True)

    # Split
    split_train_val(csv_file_path, 0.15)

def train_val_split(path, dimensions):
    images_folder_path = path + "/" + "Train"
    masks_folder_path = images_folder_path + "_GT_masks"
    val_imgs_folder_path = path + "/" + "Validation"
    val_masks_folder_path = val_imgs_folder_path + "_GT_masks"
    resize_dimensions = (dimensions)

    train_X = turn_np(images_folder_path, resize_dimensions)
    train_y = turn_np(masks_folder_path, resize_dimensions)

    test_X = turn_np(val_imgs_folder_path, resize_dimensions)
    test_y = turn_np(val_masks_folder_path, resize_dimensions)

    return train_X, train_y, test_X, test_y