#from unet import unet
from unetbn import unet
import metrics
import data_process
from PIL import Image


from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.metrics import MeanIoU

path = "D:/Users/imbrm/ISIC_2017/check/new/all"
#path = "D:/Users/imbrm/ISIC_2017-2"

bs = 16
ep = 100
size = (256,256,1)
version = 1

weights_filename = 'unet_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=2, min_delta=1e-4)
#, mode='max'
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
tensorboard = TensorBoard(log_dir='./logs/logs_1e-4')


print("Generating Dataset:")
#data_process.generate_dataset(path, size, 8)
train_X, train_y, val_X, val_y, train_y1, val_y1 = data_process.train_val_split(path, size)

print("Building model:")
model = unet(input_size=size)

print("Training model:")
history = model.fit(train_X,
                    train_y,
                    batch_size=bs,
                    validation_data = (val_X, val_y1),
                    epochs=ep,
                    callbacks=[checkpoint, reduce_lr, early_stopping, tensorboard]
                    )

model.save('model_version_{}'.format(version))

