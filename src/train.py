import os

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

import api
import argparse

parser = argparse.ArgumentParser(
    description='Recorder for training data.')
parser.add_argument('model_file', type=str, help='model_file')
parser.add_argument('num_classes', type=int, help='model_file')
parser.add_argument('data_dir', type=str, help='model_file')
parser.add_argument('--image_size', type=int, default=64, help='image_size')
parser.add_argument('--num_epochs', type=int, default=10, help='num_epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
args = parser.parse_args()

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model = api.create_model(args.image_size, args.num_classes)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
x_train, y_train = api.load_images_masks(os.path.join(args.data_dir, "train"), image_size=args.image_size)
print(x_train.shape)
print(y_train.shape)
x_test, y_test = api.load_images_masks(os.path.join(args.data_dir, "val"), image_size=args.image_size)
print(x_test.shape)
print(y_test.shape)
model_checkpoint = ModelCheckpoint(filepath=args.model_file, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)
model.fit(x_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(x_test, y_test),
          verbose=1, callbacks=[early_stopping, model_checkpoint])

model.save_weights(args.model_file)
