import copy
import numpy as np

# generate a tweet prediction from the model
def generate_tweet(model, start, max_length, temperature, end_index):
    seq = copy.copy(start)
    while (len(seq) < max_length) & (seq[-1] != end_index):
        preds = model.predict(np.array([seq]))[-1][-1] * .999
        preds = np.log(preds) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        choice = np.random.multinomial(1, preds)
        nextword = np.argmax(choice)
        seq.append(nextword)
    if seq[-1] != end_index:
        seq.append(end_index)
    return seq
    
# calculate loss of model on validation set
def get_loss(model, X_validate, Y_validate):
    loss = 0.0
    for length_index in range(0, len(X_validate)):
        # check whether there are tweets of this length at all
        if len(X_validate[length_index] != 0):
            X_batch = X_validate[length_index]
            Y_batch = Y_validate[length_index]
            # expand to use keras sparse_categorical_crossentropy
            Y_batch = np.expand_dims(Y_batch, -1)
            loss += model.evaluate(X_batch, Y_batch, verbose=0)
    return loss
    
def split_samples(X_samples, Y_samples, validation_split, train_on_all=False):
    # initialize lists for training and validation data
    X_train = []
    Y_train = []
    X_validate = []
    Y_validate = []

    # for each tweet length reserve some data for validation
    for length_index in range(0, len(X_samples)):
        tweets_of_length = len(X_samples[length_index])
        # choose validation split
        split_index = int(np.floor(validation_split * tweets_of_length))
        # split into training and validation set (train on all if flag is set)
        if train_on_all:
            X_train.append(X_samples[length_index])
            Y_train.append(Y_samples[length_index])
        else:
            X_train.append(X_samples[length_index][:split_index])
            Y_train.append(Y_samples[length_index][:split_index])
        X_validate.append(X_samples[length_index][split_index:])
        Y_validate.append(Y_samples[length_index][split_index:])
        
    return (X_train, Y_train, X_validate, Y_validate)