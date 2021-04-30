# def move_data(list, path, data_type):
#     """Move the images whose ID is in "list" from the Training folder to Validation,
#     or, in the case of masks, from Training_GT_masks to Validation_GT_masks.

#     Args:
#         list (list): List containing the IDs of validation images/masks.
#         path (string): Path to parent folder.
#         data_type (string): Defines whether it's an image or a mask.
#     """
#     input_folder = path + "/" + "Train"
#     output_folder = path + "/" + "Validation"

#     if data_type.capitalize() == "Image":
#         for image_id in list:
#             shutil.move(
#                 input_folder + "/" + image_id + ".png",
#                 output_folder + "/" + image_id + ".png",
#             )

#     if data_type.capitalize() == "Mask":
#         input_folder = input_folder + "_GT_masks"
#         output_folder = output_folder + "_GT_masks"
#         for image_id in list:
#             shutil.move(
#                 input_folder + "/" + image_id + ".png",
#                 output_folder +  "/" + image_id + ".png"
#             )


# def split(df, result, val_ratio, csv_file_path):
#     """Performs the split into train and validation data.
#     Stores the indices of images with or without melanoma in the "train" list,
#     then it moves a certain percentage of them (specified by val_ratio) into
#     the "validation" list. Finally, marks each image with "T" or "V" appropriately.
#     The split is randomly performed using random.sample() function.

#     Args:
#         df (DataFrame): Pandas DataFrame containing information about the dataset.
#         result (int): Whether it's melanoma (1) or no melanoma (0).
#         val_ratio (float): Percentage of data to be split into validation.
#         csv_file_path (string): File path for extrapolating parent path.

#     Returns:
#         df (DataFrame): The transformed DataFrame with marked images.
#     """
#     train = list(df[df["melanoma"] == result].index)
#     # ceil function for rounding up float numbers
#     n_val = math.ceil(val_ratio * len(train))
#     validation = random.sample(train, n_val)
#     validation.sort()
#     for element in validation:
#         train.pop(train.index(element))

#     # Mark validation images with "V"
#     for id in tqdm(validation):
#         df.at[id, "split"] = "V"
#     # Mark train images with "T"
#     for id in tqdm(train):
#         df.at[id, "split"] = "T"
#     val_ids = [df.at[index, "image_id"] for index in validation]
#     val_masks_ids = [df.at[index, "image_id"] + "_segmentation" for index in validation]

#     path = os.path.split(csv_file_path)[0]

#     # Move validation images to folder.
#     move_data(val_ids, path, "Image")

#     # Move validation masks to folder.
#     move_data(val_masks_ids, path, "Mask")

#     return df


# def split_train_val(csv_file_path, percent):
#     """Callable function for splitting the dataset into train and validation.

#     Args:
#         csv_file_path (string): Path to .csv file containing ground truth.
#     """
#     csv_name = splitext(os.path.basename(csv_file_path))[0]
#     csv_copy_path = os.path.split(csv_file_path)[0] + "/" + csv_name + "_split.csv"
#     shutil.copy2(csv_file_path, csv_copy_path)

#     csv_copy = pd.read_csv(csv_copy_path)
#     csv_copy["split"] = ""

#     print("Splitting the dataset into Train/Validation:")

#     # MELANOMA YES (result=1)
#     csv_copy = split(csv_copy, 1, percent, csv_file_path)
#     # MELANOMA NO (result=0)
#     csv_copy = split(csv_copy, 0, percent, csv_file_path)

#     csv_copy.to_csv(csv_copy_path, index=False)



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

def train_val_split(path):
    images_folder_path = path + "/" + "Train"
    masks_folder_path = images_folder_path + "_GT_masks"
    val_imgs_folder_path = path + "/" + "Validation"
    val_masks_folder_path = val_imgs_folder_path + "_GT_masks"

    train_X = turn_np_imgs(images_folder_path)
    train_y, train_y1 = turn_np_masks(masks_folder_path)

    val_X = turn_np_imgs(val_imgs_folder_path)
    val_y, val_y1 = turn_np_masks(val_masks_folder_path)

    return train_X, train_y, val_X, val_y, train_y1, val_y1

###################################################################################################################

def del_augm(input_path, jobs):
    """Deletes the superpixels images of the skin lesions.

    Args:
        input_path (string): Path of the folder containing all the images.
        jobs (string): Number of job for parallelisation.
    """
    # Store the IDs of all the _superpixel images in a list.
    images = [
        splitext(file)[0]
        for file in listdir(input_path)
        if "x" in splitext(file)[0]
    ]
    print("Deleting augm Images:")
    Parallel(n_jobs=jobs)(
        delayed(os.remove)(str(input_path + "/" + str(image + ".png")))
        for image in tqdm(images)
    )

###################################################################################################################

    # TRAIN
score_tr = unet.evaluate(x_train, y_train)
dice_coef_loss_tr = score_tr[0]
dice_coef_loss_tr = score_tr[1]
true_positive_tr = score_tr[2]
true_negative_tr = score_tr[3]
acc_tr = score_tr[4]

print(f"""TRAIN RESULTS: 
Train Dice Coefficient Loss: {dice_coef_loss_tr}
Train Jaccard Index Loss: {dice_coef_loss_tr}
Train True Positive: {true_positive_tr}
Train True Negative: {true_negative_tr}
Train Accuracy: {acc_tr}""")

# VALIDATION
score_val = unet.evaluate(x_val, y_val)
dice_coef_loss_val = score_val[0]
dice_coef_loss_val = score_val[1]
true_positive_val = score_val[2]
true_negative_val = score_val[3]
acc_val = score_val[4]

print(f"""RESULTS: 
Validation Dice Coefficient Loss: {dice_coef_loss_val}
Validation Jaccard Index Loss: {dice_coef_loss_val}
Validation True Positive: {true_positive_val}
Validation True Negative: {true_negative_val}
Validation Accuracy: {acc_val}""")


# TEST
score_test = unet.evaluate(x_test, y_test)
dice_coef_loss_test = score_test[0]
dice_coef_loss_test = score_test[1]
true_positive_test = score_test[2]
true_negative_test = score_test[3]
acc_test = score_test[4]

print(f"""RESULTS: 
Test Dice Coefficient Loss: {dice_coef_loss_test}
Test Jaccard Index Loss: {dice_coef_loss_test}
Test True Positive: {true_positive_test}
Test True Negative: {true_negative_test}
Test Accuracy: {acc_test}""")



# print(im.shape)

# im = im[np.newaxis,:,:,np.newaxis]

# print(im.shape)

# plt.imshow(img)
# plt.waitforbuttonpress()