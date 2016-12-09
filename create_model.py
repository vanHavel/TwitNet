import numpy as np
import argparse
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributed
from keras.layers import Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop, Adam

# parse command line arguments
parser = argparse.ArgumentParser(description="Build an RNN model.")
parser.add_argument("-o", "--output_file", default="model/model.mod", help="Path to output file for model.")
parser.add_argument("-v", "--vocab_file", default="data/vocab.txt", help="Path to vocabulary file.")
parser.add_argument("-t", "--layer_type", default="lstm", choices=["rnn","lstm","gru"], help="Type of recurrent layer to use.")
parser.add_argument("-n", "--hidden_num", default=2, type=int, help="Number of hidden layers.")
parser.add_argument("-s", "--hidden_size", default=128, type=int, help="Number of neurons per hidden layer.")
parser.add_argument("-a", "--optimizer", default="adam", choices=["adam","rmsprop"], help="Optimizer to use.")
parser.add_argument("-l", "--learning_rate", default=0.001, type=float, help="Initial learning rate for the optimizer.")
parser.add_argument("-d", "--dropout", default=0.0, type=float, help="If set to 0 < p < 1: apply dropout with p after each recurrent layer.")
args = parser.parse_args()

output_file = args.output_file
vocab_file = args.vocab_file
layer_type = args.layer_type
hidden_num = args.hidden_num
hidden_size = args.hidden_size
optimizer = args.optimizer
learning_rate = args.learning_rate
dropout = args.dropout

# check arguments
if hidden_num < 1:
    print ("Invalid hidden_num value, must be positive.")
    sys.exit(1)
if hidden_size < 1:
    print ("Invalid hidden_size value, must be positive.")
    sys.exit(1)
if learning_rate <= 0.0:
    print("Invalid learning rate, may not be negative.")
    sys.exit(1)
if dropout >= 1.0 or dropout < 0.0:
    print("Invalid dropout, must be in [0,1).")

# read vocabulary to get its size
print ("Reading vocabulary.")
vfile = open(vocab_file, "r")
vocab_size = len(vfile.readlines())
vfile.close()

# build the keras model
print("Building model.")
model = Sequential()
# embedding layer, only add if vocab_size is significantly larger than hidden size
if vocab_size > (hidden_size * 5):
    model.add(Embedding(vocab_size, hidden_size))
    layer_start = 0
else:
    # start directly with the first layer, and specify input shape
    if layer_type == "rnn":
        model.add(SimpleRNN(output_dim=hidden_size, return_sequences=True, input_shape=(None, vocab_size)))
    elif layer_type == "lstm":
        model.add(LSTM(output_dim=hidden_size, return_sequences=True, input_shape=(None, vocab_size)))
    elif layer_type == "gru":
        model.add(GRU(output_dim=hidden_size, return_sequences=True, input_shape=(None, vocab_size)))
    # maybe add dropout
    if (dropout > 0.0):
        model.add(Dropout(dropout))
    layer_start = 1
# recurrent layers
for i in range(layer_start, hidden_num):
    if layer_type == "rnn":
        model.add(SimpleRNN(output_dim=hidden_size, return_sequences=True))
    elif layer_type == "lstm":
        model.add(LSTM(output_dim=hidden_size, return_sequences=True))
    elif layer_type == "gru":
        model.add(GRU(output_dim=hidden_size, return_sequences=True))
    # maybe add dropout
    if (dropout > 0.0):
        model.add(Dropout(dropout))

# end with softmax operation
model.add(TimeDistributed(Dense(output_dim=vocab_size)))
model.add(Activation("softmax"))

# choose optimizer and compile
if optimizer == "rmsprop":
    model_optimizer = RMSprop(lr=learning_rate);
elif optimizer == "adam":
    model_optimizer = Adam(lr=learning_rate)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=model_optimizer)
              
# print summary
print (model.summary())

# save model
print ("Saving model to %s." % output_file)
model.save(output_file)