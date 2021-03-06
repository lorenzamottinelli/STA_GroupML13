import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from keras_visualizer import visualizer



# generate data set
t_start = 0
t_end = 10
t_steps = 10
t_range = np.linspace(t_start,t_end,t_steps)
seq = np.array([np.sin(2*np.pi*t/10) for t in t_range])
[print(num) for num in enumerate(seq)]
#In order to create training data for predicting the next value of the sine function based on the current value,
#generate n pairs of two-point sequences from the values in seq.
n_training_data_pairs = 100
x_train = np.array([])
y_train = np.array([])
for i in range(n_training_data_pairs):
    rand = np.random.randint(t_steps)
    x_train = np.append(x_train, seq[rand])
    y_train = np.append(y_train, seq[np.mod(rand + 1,t_steps)])
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
        #layers.Dense(20, activation='relu'),
        #layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])
    return model
model = build_model()
model.summary()
plot_model(model, to_file='model_visual.png')
visualizer(model,format='png', view=True)

#train model
history = model.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=100)

#Review the training progress and performance
#View the model's (historical) training progress via the history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#plot training history
plt.plot(history.history['loss'], color='xkcd:blue')
plt.plot(history.history['val_loss'], color='xkcd:red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss'],loc='upper right')
plt.show()

#predict values using data in the test set
y_pred = model.predict(x_test, batch_size=10).flatten()
a = plt.axes(aspect='equal')
plt.scatter(y_test, y_pred, color='blue')
_ = plt.plot([-2,2], [-2,2], color='xkcd:light grey')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.show()

#Visualize the predicted value versus the training (ie. true) values.
plot_test = plt.plot(t_range, y_test,'x-', color='xkcd:blue', label='y_test')
plot_pred = plt.plot(t_range,y_pred,'o', color='xkcd:orange', label='y_pred')
plt.xlabel('x')
plt.ylabel('sin(x)')
test_patch = mpatches.Patch(color='xkcd:blue', label='test data')
pred_patch = mpatches.Patch(color='xkcd:orange', label='pred data')
plt.legend(handles = [test_patch, pred_patch])
plt.show()

#Take a look at the error distribution.
error = y_pred - y_test
plt.hist(error, bins = 25, color='xkcd:blue')
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")

# Normalize
