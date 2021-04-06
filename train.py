import unet
import metrics
import data_process

from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

path = "D:/Users/imbrm/ISIC_2017"
#path = "D:/Users/imbrm/ISIC_2017-2/check"
size = (572,572,1)
version = 0

early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=2)
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_competition_metric', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
#optimizer = SGD(lr=0.1, momentum=0.8, nesterov=False)
optimizer = Adam(lr=0.1)

print("Generating Dataset:")
data_process.generate_dataset(path, size, 1)
train_X, train_y, val_X, val_y = data_process.train_val_split(path, size)
print("Building and compiling model.")
model = unet.build_model(size, True)
model.compile(optimizer=optimizer, loss=metrics.iou_bce_loss, metrics=['accuracy', metrics.jaccard_distance_loss, metrics.dice_loss, metrics.focal_loss])
print("Training model.")
history = model.fit(train_X,
                    train_y,
                    batch_size=64,
                    validation_data = [val_X, val_y],
                    epochs=5,
                    callbacks=[checkpoint, reduce_lr]
                    )