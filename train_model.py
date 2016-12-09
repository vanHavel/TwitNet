import numpy as np
import sys
import os
import argparse
from datetime import *

import utility.model

from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding

# parse command line arguments
parser = argparse.ArgumentParser(description="Train an RNN model.")
parser.add_argument("-m", "--model_file", default="model/model.mod", help="Path to model file.")
parser.add_argument("-v", "--vocab_file", default="data/vocab.txt", help="Path to vocabulary file.")
parser.add_argument("-t", "--training_file", default="data/training_data.npz", help="Path to training data file.")
parser.add_argument("-e", "--epochs", default=20, type=int, help="Number of epochs to train.")
parser.add_argument("-b", "--batchsize", default=32, type=int, help="Minibatch size for training.")
parser.add_argument("-s", "--save_every", default=1, type=int, help="Save model paramters after every n epochs.")
parser.add_argument("-l", "--evaluate_every", default=1, type=int, help="Evaluate loss after every n epochs.")
parser.add_argument("-p", "--validation_split", default=0.1, type=float, help="Part of training data used as validation data for evaluating loss.")
parser.add_argument("-a", "--train_on_all", action="store_true", help="Train on all examples, including the validation samples. Might lead to overfitting.")
args = parser.parse_args()

model_file = args.model_file
vocab_file = args.vocab_file
training_file = args.training_file
epochs = args.epochs
batchsize = args.batchsize
save_every = args.save_every
evaluate_every = args.evaluate_every
validation_split = args.validation_split
train_on_all = args.train_on_all

# check arguments
if epochs < 1:
    print ("Invalid epoch value, must be positive.")
    sys.exit(1)
if batchsize < 1:
    print ("Invalid batchsize value, must be positive.")
    sys.exit(1)
if save_every < 1:
    print ("Invalid save_every value, must be positive.")
    sys.exit(1)
if evaluate_every < 1:
    print ("Invalid evaluat_every value, must be positive.")
    sys.exit(1)
if validation_split <= 0.0 or validation_split >= 1.0:
    print("Invalid validation split, must be in (0,1).")
    sys.exit(1)

# read training data
print("Reading training data.")
data = np.load(training_file)
X_samples = data['X_samples']
Y_samples = data['Y_samples']

# split into training and validation data
(X_train, Y_train, X_validate, Y_validate) = utility.model.split_samples(X_samples, Y_samples, validation_split, train_on_all)

# read vocabulary
print ("Reading vocab.")
vfile = open(vocab_file, "r")
index_to_word = [line[:-1] for line in vfile.readlines()]
vocab_size = len(index_to_word)
word_to_index = dict([(c,i) for i,c in enumerate(index_to_word)])
vfile.close()

# load the model            
print ("Loading model.")
model = load_model(model_file)

# generate output path for saved models
(model_dirname, model_basename) = os.path.split(model_file)
(filename, extension) = os.path.splitext(model_basename)
model_basepath = os.path.join(model_dirname, filename)

# get starting loss of model
print ("Evaluating loss.")
current_loss = utility.model.get_loss(model, X_validate, Y_validate)
print ("Loss before training: %f." % current_loss)

# training loop
print ("Training model.")
for current_epoch in range(1, epochs + 1):
    # print epoch number and timestamp
    timestamp = str(datetime.now())
    print ("%s: Epoch %d" % (timestamp, current_epoch))
    
    # iterate over tweet lengths
    for length_index in range(0, len(X_train)):
        # check whether there are tweets of this length at all
        if len(X_train[length_index] != 0):
            X_batch = X_train[length_index]
            Y_batch = Y_train[length_index]
            # randomly permute tweets of the given length
            p = np.random.permutation(len(X_batch))
            X_batch = X_batch[p]
            Y_batch = Y_batch[p]
            # expand dimension to use keras sparse_categorical_crossentropy
            Y_batch = np.expand_dims(Y_batch, -1)
            # if there is no embeddding layer: transform X_batch to one-hot-encoding
            if not (type(model.layers[0]) is Embedding):
                X_batch = to_categorical(X_batch, vocab_size)
            # train network for 1 epoch on the batch
            model.fit(X_batch, Y_batch, batch_size=batchsize, nb_epoch=1, verbose=0)
    
    # calculate loss and adjust learning rate if necessary
    if (current_epoch % evaluate_every) == 0:
        old_loss = current_loss
        current_loss = utility.model.get_loss(model, X_validate, Y_validate)
        print("Loss after epoch %d: %f." % (current_epoch, current_loss))
        if old_loss < current_loss:
            # loss has increased, learning rate is halved
            learning_rate = model.optimizer.lr.get_value()
            learning_rate /= 2
            model.optimizer.lr.set_value(learning_rate)
            print("Adjusting learning rate to %f." % learning_rate)
    
    # save model if necessary        
    if (current_epoch % save_every) == 0:
        print ("Saving model after epoch %d" % current_epoch)
        model.save(model_basepath + "_epoch" + str(current_epoch) + ".mod")

# save model in the end
print ("Saving model after %d epochs." % epochs)
model.save(model_basepath + "_epoch" + str(current_epoch) + ".mod")