import arg_parse
import keras_tuner
from model import toy_model


parser = arg_parse.get_args()
args = parser.parse_args()
print(args.model_type)
toy_model(keras_tuner.HyperParameters())


