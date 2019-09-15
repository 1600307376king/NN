from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

iris_data = load_iris()
print(iris_data.keys())

y = iris_data.target
x = iris_data.data[y < 2, :3]
labels = y[:100]

train_x, test_x, train_y, test_y = train_test_split(x, labels, test_size=0.25, random_state=42)

model = keras.Sequential([
                          keras.layers.Dense(128, activation=tf.nn.sigmoid)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=500)

test_loss, test_acc = model.evaluate(test_x, test_y)
print(test_acc)