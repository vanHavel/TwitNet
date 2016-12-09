import copy
import numpy as np

from keras.layers import Embedding
from keras.utils.np_utils import to_categorical

def softmax(preds):
    log_softmax = preds - np.log(np.sum(np.exp(preds)))
    return np.exp(log_softmax)

# sample a new word from a given sequence
def sample(model, seq, temperature, vocab_size):
    if not (type(model.layers[0]) is Embedding):
        # transform to one-hot encoding
        net_input = [to_categorical(seq, vocab_size)]
    else:
        net_input = [seq]
    # predict from model
    preds = model.predict(np.array(net_input))[-1][-1]
    # rescale with temperature
    preds = np.log(preds) / temperature
    preds = softmax(preds)
    # multiply with .99 to be save below 1
    preds *= .99
    # sample from distribution
    choice = np.random.multinomial(1, preds)
    return np.argmax(choice)

# generate a tweet prediction from the model
def generate_tweet(model, start, end_index, vocab_size, unknown_index=0, max_length=30, temperature=1.0, sample_unknown=False):
    seq = copy.copy(start)
    while (len(seq) < max_length) & (seq[-1] != end_index):
        nextword = unknown_index
        if sample_unknown:
            nextword = sample(model, seq, temperature, vocab_size)
        else:
            while (nextword == unknown_index):
                nextword = sample(model, seq, temperature)
        seq.append(nextword)
    if seq[-1] != end_index:
        seq.append(end_index)
    return seq

# transfrom batch to one hot encoding    
def to_one_hot(X_batch, vocab_size):
    new_batch = np.zeros((len(X_batch), len(X_batch[0]), vocab_size))
    for i in range(0, len(X_batch)):
        new_batch[i] = to_categorical(X_batch[i], vocab_size)
    return new_batch
    
# calculate loss of model on validation set
def get_loss(model, X_validate, Y_validate, vocab_size):
    loss = 0.0
    for length_index in range(0, len(X_validate)):
        # check whether there are tweets of this length at all
        if len(X_validate[length_index] != 0):
            X_batch = X_validate[length_index]
            Y_batch = Y_validate[length_index]
            loss += model.evaluate(X_batch, Y_batch, verbose=0)
    return loss
    
def split_samples(model, X_samples, Y_samples, validation_split, vocab_size, train_on_all=False):
    # initialize lists for training and validation data
    X_train = []
    Y_train = []
    X_validate = []
    Y_validate = []

    # for each tweet length reserve some data for validation
    for length_index in range(0, len(X_samples)):
        tweets_of_length = len(X_samples[length_index])
        X_batch = X_samples[length_index]
        Y_batch = Y_samples[length_index]
        # if there is no embeddding layer: transform X_batch to one-hot-encoding
        if not (type(model.layers[0]) is Embedding):
            X_batch = to_one_hot(X_batch, vocab_size)
        # expand to use keras sparse_categorical_crossentropy
        Y_batch = np.expand_dims(Y_batch, -1)
        # choose validation split
        split_index = int(np.floor(validation_split * tweets_of_length))
        # split into training and validation set (train on all if flag is set)
        if train_on_all:
            X_train.append(X_batch)
            Y_train.append(Y_batch)
        else:
            X_train.append(X_batch[:split_index])
            Y_train.append(Y_batch[:split_index])
        X_validate.append(X_batch[split_index:])
        Y_validate.append(Y_batch[split_index:])
        
    return (X_train, Y_train, X_validate, Y_validate)