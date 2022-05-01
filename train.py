#import libraries
from gc import callbacks
import tensorflow as tf
print("TensorFlow version:" + tf.__version__)
from src import myModel
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from  src import reportGenerator
# set trained model save path
save_path = "./saved_models/"
name = "myModel"




# prepare dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train/255, X_test/255

# build and compile model
model_creator = myModel.dnnModel()
model = model_creator.create((28, 28), 10)
print(model.predict(X_train[:100]).shape)
model.summary()


#fit model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 1000, batch_size = 64, callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)])

report_data = history.history

reportGenerator.generate(report_data)


model.save(save_path + name + ".h5")