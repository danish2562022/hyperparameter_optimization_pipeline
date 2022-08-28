import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import arg_parse

parser = arg_parse.get_args()
opt = parser.parse_args()
class CustomTuning(keras_tuner.HyperModel):

    def build(self,hp):

        model = keras.Sequential()
        model.add(layers.Flatten())
        
        for i in range(hp.Int("num_layers",1,opt.max_number_of_layers)):
       
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Int(f"units_{i}", min_value=opt.min_units_per_layers, max_value=opt.max_units_per_layers, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            if hp.Boolean(f"dropout_{i}"):
                model.add(layers.Dropout(rate=0.25))

        if opt.model_type == 'r':
            model.add(layers.Dense(1))
        else:
            model.add(layers.Dense(opt.num_of_classes, activation="softmax"))

        return model



    def fit(self, hp, model, x_train, y_train, validation_data, callbacks=None, **kwargs):

        batch_size = hp.Int("batch_size", 32, 128, step=32, default=64)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(
            batch_size
        )

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        if opt.choose_optimizer == 'adam':
            optim = keras.optimizers.Adam(learning_rate=learning_rate)

        elif opt.choose_optimizer == 'sgd':
            optim = keras.optimizers.SGD(learning_rate=learning_rate)

        if opt.model_type == 'r':
            loss_fn =  tf.keras.losses.MeanAbsoluteError()
        else:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

        epoch_loss_metric = keras.metrics.Mean()



        @tf.function
        def run_train_step(features,labels):

            with tf.GradientTape() as tape:
                pred_labels = model(features)
                loss = loss_fn(labels, pred_labels)

                if model.losses:
                    loss += tf.math.add_n(model.losses)



        






        

