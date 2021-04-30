from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from metrics import dice_coef_loss, jaccard_coef_loss, true_positive, true_negative
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = ResNet50()

model.compile(loss=dice_coef_loss, optimizer='Adam', metrics=[jaccard_coef_loss, true_positive, true_negative])

x_train = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/data.npy')
y_train = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataMask.npy')

x_test = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataval.npy')
y_test = np.load('D:/Users/imbrm/ISIC_2017-2/hehesmall/dataMaskval.npy')



checkpoint = ModelCheckpoint("/var/tmp/mi714/aug17/models/resnet/unet_bn.h5", 
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'
                             )
early_stopping = EarlyStopping(patience=10,
                               monitor='val_loss',
                               mode='min',
                               verbose=1,
                               min_delta=0.01
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              min_delta=0.01,
                              cooldown=0,
                              min_lr=0.5e-7,
                              factor=0.5,
                              patience=5,
                              verbose=1
                              )

history = model.fit(trainData, 
                    trainMask,
                    epochs=50,
                    batch_size = 5,
                    validation_data=(valData, valMask),
                    verbose=2)


## TAKEN FROM 
# https://github.com/bnsreenu/python_for_microscopists/blob/master/203b_skin_cancer_lesion_classification_V4.0.py
# # Prediction on test data
# y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors 
# y_pred_classes = np.argmax(y_pred, axis = 1) 
# # Convert test data to one hot vectors
# y_true = np.argmax(y_test, axis = 1) 

# #Print confusion matrix
# cm = confusion_matrix(y_true, y_pred_classes)

# fig, ax = plt.subplots(figsize=(6,6))
# sns.set(font_scale=1.6)
# sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


# #PLot fractional incorrect misclassifications
# incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
# plt.bar(np.arange(7), incorr_fraction)
# plt.xlabel('True Label')
# plt.ylabel('Fraction of incorrect predictions')


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('D:/Users/imbrm/ISIC_2017-2/loss.png')
# summarize history for jaccard
plt.plot(history.history['jaccard_coef_loss'])
plt.plot(history.history['val_jaccard_coef_loss'])
plt.title('model jaccard coef loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('D:/Users/imbrm/ISIC_2017-2/jaccard.png')
# summarize history for positive
plt.plot(history.history['true_positive'])
plt.plot(history.history['val_true_positive'])
plt.title('model true positive')
plt.ylabel('positives')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('D:/Users/imbrm/ISIC_2017-2/pos.png')
# summarize history for negative
plt.plot(history.history['true_negative'])
plt.plot(history.history['val_true_negative'])
plt.title('model true negative')
plt.ylabel('negatives')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('D:/Users/imbrm/ISIC_2017-2/neg.png')


score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])


# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]