import numpy as np
import random
import string
import copy

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
    
        
# return a random sequnce of n letters or digits
def random_chars(n):
    chars = [random.choice(string.ascii_letters + string.digits) for i in range(0,n)]
    return "".join(chars)
      
# postprocess tweets, fixing hashtags, links etc.       
def postprocess(seq):
    # cutoff start and end
    seq = seq[1:-1]    
    # join to string
    seq = " ".join(seq)
    #postprocess
    seq = seq.replace("@ ", "@")
    seq = seq.replace("# ", "#")
    seq = seq.replace("http : ", "http:")
    seq = seq.replace("https : ", "https:")
    seq = seq.replace("<link>", "tinyurl.com/" + random_chars(8))
    seq = seq.replace("amp ;", "&")
    seq = seq.replace(" 's", "'s")
    seq = seq.replace(" 're", "'re")
    seq = seq.replace(" n't", "n't")
    seq = seq.replace(" 'm", "'m")
    seq = seq.replace(" !", "!")
    seq = seq.replace(" .", ".")
    seq = seq.replace(" amp ", " & ")
    return seq