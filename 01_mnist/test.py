import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from model import create_model


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

# Testing
model = create_model()
checkpoint_path = "./training_1/cp.ckpt"#

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))



