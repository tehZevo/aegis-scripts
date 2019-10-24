import argparse
import tensorflow as tf

from ml_utils.model_builders import dense_autoencoder

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-shape", nargs="+", type=int, required=True)
parser.add_argument("-l", "--latent-size", type=int, required=True)
parser.add_argument("-s", "--hidden-sizes", nargs="+", type=int, default=[])
parser.add_argument("-a", "--activation", type=str, default="tanh")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-4)
parser.add_argument("-o", "--optimizer", type=str, default="adam")
parser.add_argument("-f", "--filepath", type=str, required="true")
#TODO: loss arg
args = parser.parse_args()

#TODO: support more optimizers by name... or by object
optimizer = None if args.optimizer is None else tf.keras.optimizers.Adam(args.learning_rate)

#create autoencoder
model = dense_autoencoder(args.input_shape[0], args.latent_size, args.hidden_sizes,
  acti=args.activation)
model.compile(loss="mse", optimizer=optimizer)
print(model.summary())

model.save(args.filepath)
