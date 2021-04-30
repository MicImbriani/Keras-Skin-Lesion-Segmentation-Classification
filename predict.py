#import matplotlib.pyplot as plt
import os
import random 
import numpy as np
from keras.models import load_model

path = "D:/Users/imbrm/ISIC_2017/check/new/all"
input_size = (256,256,1)

#unet = unet(input_size)
#unet_weights = ''
#unet.load_weights(unet_weights)



# img = Image.open("D:/Users/imbrm/ISIC_2017/check/new/all/Validation/ISIC_0000034.png")
# im = cv2.imread("D:/Users/imbrm/ISIC_2017/check/new/all/Validation/ISIC_0000034.png", 0)

def get_results(path, model):
    model = model
    x_train = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/datatest.npy')
    y_train = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataMasktest.npy')

    x_val = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/datatest.npy')
    y_val = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataMasktest.npy')

    x_test = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/datatest.npy')
    y_test = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataMasktest.npy')

    # RESULTS
    sets = [(x_train, y_train),
            (x_val, y_val),
            (x_test, y_test)]
    for x, y in sets:
        score = model.evaluate(x, y)
        dice_coef_loss = score[0]
        dice_coef_loss = score[1]
        true_positive = score[2]
        true_negative = score[3]
        acc = score[4]

        print(f"""RESULTS: 
        Dice Coefficient Loss: {dice_coef_loss}
        Jaccard Index Loss: {dice_coef_loss}
        True Positive: {true_positive}
        True Negative: {true_negative}
        Accuracy: {acc}""")




def get_prediction():
    x_train = ""
    y_train = ""

    x_val = ""
    y_val = ""

    x_test = ""
    y_test = ""

    folder_sets = [(x_train, y_train),
                (x_val, y_val),
                (x_test, y_test)]

    for img, mask in folder_sets:
        files = os.listdir(path)
        im = random.choice(files)

        x = Image.open(img + "/" + im + ".png")
        y = Image.open(mask + "/" + mask + ".png")

        y_pred = model.predict(x)

        image_ = np.array(x)
        image_ = image_.astype(np.float64)
        mask_ = np.array(y)
        mask_ = mask_.astype(np.float64)
        maskpred = np.array(y_pred)
        maskpred = maskpred.astype(np.float64)

        plt.imshow(image_)
        plt.waitforbuttonpress()
        plt.imshow(mask_)
        plt.waitforbuttonpress()
        plt.imshow(maskpred)
        plt.waitforbuttonpress()
