from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
from model import create_model

tf.__version__



# Loading MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels = train_labels[:60000]
test_labels = test_labels[:10000]


img_rows, img_cols = 28, 28
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
print("test")
print(train_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Setting check-point path
checkpoint_path = "/home/csk/Documents/02_safe_AI/04_mnist/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)
# Training
# Creating model
model = create_model(temp = 1.0)
#model.fit(train_images, train_labels, batch_size=32, epochs=10)
# Training model
model.fit(train_images, train_labels, batch_size=32, epochs=5,
        validation_data = (test_images, test_labels), 
        callbacks = [cp_callback])  # pass callback to training

          

