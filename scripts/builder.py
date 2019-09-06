import argparse
import tensorflow as tf

from ml_utils.model_builders import dense_stack

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-shape", nargs="+", type=int, required=True)
parser.add_argument("-o", "--output-shape", nargs="+", type=int, required=True)
parser.add_argument("-s", "--hidden-sizes", nargs="+", type=int, default=[])
parser.add_argument("-a", "--activation", type=str, default="tanh")
parser.add_argument("-A", "--output-activation", type=str, default="tanh")
parser.add_argument("-r", "--rnn", type=str, default="")
parser.add_argument("-f", "--filepath", type=str, required="true")
args = parser.parse_args()

rnn = None if args.rnn is None \
  else tf.keras.layers.LSTM if args.rnn.lower() == "lstm" \
  else tf.keras.layers.GRU if args.rnn.lower() == "gru" \
  else tf.keras.layers.SimpleRNN if args.rnn.lower() == "simple" else None

model = dense_stack(args.input_shape[0], args.output_shape[0], args.hidden_sizes,
  rnn, args.activation, args.output_activation)

print(model.summary())

model.save(args.filepath)
