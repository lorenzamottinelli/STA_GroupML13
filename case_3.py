#!/usr/bin/env python
# coding: utf-8

# # ST>A | Special Topics in Astrodynamics | ae4889

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras_visualizer import visualizer
from IPython.core.display import Image, display

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.spines.top'] = plt.rcParams['axes.spines.right'] = False
np.set_printoptions(precision=4, suppress=True)

JULIAN_DAY = 86400.0
launch_window22 = 8304.5 * JULIAN_DAY

departure_range = np.array([-25, 24]) * JULIAN_DAY + launch_window22
arrival_range = np.array([-25 + 60, 24 + 449]) * JULIAN_DAY + launch_window22
time_of_flight_range = arrival_range - departure_range

raw_data = np.loadtxt('Output.txt', skiprows=1, delimiter=',')
time_of_flight = raw_data[:, 1] - raw_data[:, 0]

raw_data = np.vstack((raw_data[:, 0], time_of_flight, raw_data[:, 1], raw_data[:, 2], raw_data[:, 3])).T


column_names = ['Departure time [s]', 'Time of flight [s]', 'Arrival time [s]', '∆V_1 [m/s]', '∆V_2 [m/s]']
raw_dataset = pd.DataFrame(data=raw_data, columns=column_names)


def do_run(sample_size, batch_size, n_nodes, n_layers):
    dataset = raw_dataset.copy()[:sample_size]

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # ## Normalisation

    def normalise_dataset(dataset):
        for i, the_range in zip(column_names, [departure_range, time_of_flight_range, arrival_range]):
            dataset[i] = (dataset[i] - the_range.min()) / (the_range.max() - the_range.min())

        for column in column_names[3:]:
            column_stats = dataset[column].describe()
            dataset[column] = (dataset[column] - column_stats.loc['mean']) / column_stats.loc['std']

        return dataset

    train_data_normalised = normalise_dataset(train_dataset.copy())

    model = keras.Sequential([
        layers.Dense(n_nodes, activation='relu', input_shape=[2]),
        *[layers.Dense(n_nodes, activation='relu') for _ in range(n_layers - 1)],
        layers.Dense(2)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    visualizer(model, format='png')

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        min_delta=0.0005,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=False
    )

    history = model.fit(train_data_normalised[[column_names[0], column_names[1]]].to_numpy(),
                        train_data_normalised[[column_names[3], column_names[4]]].to_numpy(),
                        validation_split=0.2,
                        epochs=1000,
                        batch_size=batch_size,
                        callbacks=[early_stop], verbose=0)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    test_data_normalised = normalise_dataset(test_dataset.copy())
    test_data_input = test_data_normalised[[column_names[0], column_names[1]]].to_numpy()
    test_data_labels = test_data_normalised[[column_names[3], column_names[4]]].to_numpy()
    prediction = model.predict(test_data_input)
    evaluation = model.evaluate(test_data_input, test_data_labels, verbose=0)

    errors = pd.DataFrame(data=np.vstack((test_data_normalised[column_names[3]] - prediction[:, 0], test_data_normalised[column_names[4]] - prediction[:, 1])).T,
                          columns=["norm[∆V_1] error", "norm[∆V_2] error"])

    def print_data(data):
        mean, std = np.average(data), np.std(data)
        print(mean, std, mean + std, end=' ', sep=' ')

    # print('sample_size', 'n_layers', 'n_nodes', 'batch_size')
    print(sample_size, n_layers, n_nodes, batch_size, end=' ')
    print_data(errors["norm[∆V_1] error"])
    print_data(errors["norm[∆V_2] error"])
    print(*evaluation)


for n_layers in [5, 4, 3, 2, 10, 1]:
    sample_size = 100_000
    batch_size = sample_size // 20
    n_nodes = 50
    do_run(sample_size=sample_size, batch_size=batch_size, n_nodes=n_nodes, n_layers=n_layers)

