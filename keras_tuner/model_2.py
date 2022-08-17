from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import tensorflow as tf
import numpy as np


def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int("conv_layers", 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=hp.Int("filters_" + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int("kernel_size_" + str(i), 3, 5),
            activation="relu",
            padding="same",
        )(x)

        if hp.Choice("pooling" + str(i), ["max", "avg"]) == "max":
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice("global_pooling", ["max", "avg"]) == "max":
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice("optimizer", ["adam", "sgd"])
    model.compile(
        optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


tuner = keras_tuner.Hyperband(
    hypermodel=build_model,
    objective="val_accuracy",
    max_epochs=2,
    factor=3,
    hyperband_iterations=1,
    distribution_strategy=tf.distribute.MirroredStrategy(),
    directory="results_dir",
    project_name="mnist",
    overwrite=True,
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the images to have the channel dimension.
x_train = (x_train.reshape(x_train.shape + (1,)) / 255.0)[:1000]
y_train = y_train.astype(np.int64)[:1000]
x_test = (x_test.reshape(x_test.shape + (1,)) / 255.0)[:100]
y_test = y_test.astype(np.int64)[:100]

tuner.search(
    x_train,
    y_train,
    steps_per_epoch=600,
    validation_data=(x_test, y_test),
    validation_steps=100,
    callbacks=[tf.keras.callbacks.EarlyStopping("val_accuracy")],
)

