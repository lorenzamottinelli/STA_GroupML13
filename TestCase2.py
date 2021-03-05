import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# generate data set
seq = np.array([np.sin(2*np.pi*t/10) for t in range(10)])
[print(num) for num in enumerate(seq)]

#In order to create training data for predicting the next value of the sine function based on the current value,
#generate 100 pairs of two-point sequences from the values in seq.
num_sequences = 100
x_train = np.array([])
y_train = np.array([])
for i in range(num_sequences):
    rand = np.random.randint(10)
    x_train = np.append(x_train, seq[rand])
    y_train = np.append(y_train, seq[np.mod(rand + 1, 10)])
x_test = np.array(seq)
y_test = np.array(np.roll(seq, -1))
pdata = pd.DataFrame({'x':x_train,'y':y_train})
print(pdata)

#Create a scatter plot of the generated points
plt.scatter(pdata['x'],pdata['y'], color='xkcd:blue', marker='o', label='training data')
plt.title('Training data')
plt.legend()
plt.show()

# Create a Dense, Sequential model with a single layer of 10 neurons.
def build_model():
    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=[1]),
#        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])
    return model
model = build_model()
model.summary()