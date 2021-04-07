import unet
import unet2
import metrics
import data_process
import new_metrics_2 as nm
from PIL import Image


from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import MeanIoU


path = "D:/Users/imbrm/ISIC_2017/check/new/all"
#path = "D:/Users/imbrm/ISIC_2017-2"
size = (256,256,1)
version = 0


early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=2)
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_competition_metric', save_freq='epoch', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)


print("Generating Dataset:")
data_process.generate_dataset(path, size, 8)
train_X, train_y, val_X, val_y, train_y1, val_y1 = data_process.train_val_split(path, size)


# print("Building model:")
# #model = unet.build_model(input_size=size)
# model = unet2.unet(input_size=size)
# m = MeanIoU(2, name=None, dtype=None)
# print("Training model:")
# history = model.fit(train_X,
#                     train_y1,
#                     batch_size=1,
#                     validation_data = (val_X, val_y1),
#                     epochs=5,
#                     callbacks=[checkpoint, reduce_lr]
#                     )
# # import numpy as np
# # #ay = (z * 255).astype(np.uint8)
# # new_image = Image.fromarray(np.asarray(train_X[0], dtype=np.uint8))
# # new_image.save("ay.png")

# model.save('model_version{}'.format(version))

