
import arg_parse
import keras_tuner
from contextlib import redirect_stdout
from custom_model_tuning import *
from data_loader import *
import best_model



parser = arg_parse.get_args()
args = parser.parse_args()


tuner = keras_tuner.RandomSearch(
        hypermodel = CustomTuning(),
        max_trials = args.max_trials,
        overwrite = True,
        directory = "results",
        distribution_strategy=tf.distribute.MirroredStrategy(),
        project_name= "custom_training",
    )

x_train,x_test,x_val,y_train,y_test,y_val = data_loader() 
tuner.search(x_train, y_train, epochs=args.epochs, validation_data=(x_val, y_val))
best_hps = tuner.get_best_hyperparameters()[0]
with open("best_model_params.txt", "w") as external_file:
    print(best_hps.values, file = external_file)
    external_file.close()


