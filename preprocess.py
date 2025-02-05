import numpy as np
import os
import sys
import argparse
import pickle

import utility.process

tweet_start = "<tweet_start>"
tweet_end = "<tweet_end>"
unknown_token = "<unknown>"
link_token = "<link>"
number_token = "<number>"
user_token = "<username>"

# parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess tweet data for word based language model.")
parser.add_argument("-i", "--input_file", default="data/tweets.txt", help="Path to tweet input file.")
parser.add_argument("-v", "--vocab_size", default=4000, type=int, help="Size of the vocabulary.")
parser.add_argument("-m", "--min_length", default=3, type=int, help="Minimum word length of a tweet.")
parser.add_argument("-c", "--case_sensitive", action="store_true", help="If set, handle words case-sensitive.")
parser.add_argument("-u", "--tokens_unchanged", action="store_true", help="If set, do not replace individual links, usernames and numbers.")
args = parser.parse_args()

input_file = args.input_file
vocab_size = args.vocab_size
min_length = args.min_length
case_sensitive = args.case_sensitive
tokens_unchanged = args.tokens_unchanged

# check arguments
if vocab_size < 1:
    print ("Invalid vocab_size value, must be positive.")
    sys.exit(1)
if min_length < 1:
    print ("Invalid min_length value, must be positive.")
    sys.exit(1)

# create paths to output files
(drive, filename) = os.path.split(input_file)
vocab_file = os.path.join(drive, "vocab.txt")
training_file = os.path.join(drive, "training_data.npz")
data_file = os.path.join(drive, "tweet_data.pickle")

# open input file
print ("Reading input tweets.")
ifile = open(input_file, "r")
lines = ifile.readlines()
ifile.close()

# tokenize lines
print ("Preprocessing input.")
(usernames, numbers, links, lines) = utility.process.tokenize(lines, unchanged=tokens_unchanged, case=case_sensitive)
print ("Processed %d tweets." % len(lines))
        
# count word occurrences
print ("Building vocabulary of %d words." % vocab_size)
vocabdict = dict()
for line in lines:
    for word in line:
        if word not in vocabdict:
            vocabdict[word] = 1
        else:
            vocabdict[word] += 1
            
# get most frequent tokens, reserving 3 for start, end and unknown tokens
sortedvoc = sorted(vocabdict.items(), key=(lambda x: x[1]), reverse=True)
vocab = sortedvoc[:vocab_size - 3]
print ("Count of least frequent word: %d" % vocab[-1][1])

# build index to word, word to index tables
index_to_word = [x[0] for x in vocab] + [tweet_start, tweet_end, unknown_token]
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# write vocab to file
print ("Writing vocabulary to %s." % vocab_file)
ofile = open(vocab_file, "w")
for w in index_to_word:
    ofile.write(w + "\n")
ofile.close()

#process tweets
tweets = lines

print ("Replacing unknown words.")
# replace unknown words and add start/end token
for i, tweet in enumerate(tweets):
    tweets[i] = [tweet_start] + [word if word in word_to_index else unknown_token for word in tweet] + [tweet_end]
 
# Create the training data
print ("Creating training data.")
# sort tweets by length
maxlen = max([len(t) for t in tweets])
tweets = sorted(tweets, key=len)
# remove very short tweets
tweets = list(filter(lambda x: len(x) >= min_length + 2, tweets))
# group tweets by length, ignoring very short tweets
tweets = [list(filter (lambda x: len(x) == i, tweets)) for i in range(min_length + 2, maxlen)]

# map words to indices and store training sequences in list of numpy arrays. There is one array for each tweet length.
X_samples = [np.asarray([[word_to_index[w] for w in tweet[:-1]] for tweet in itweets]) for itweets in tweets]
Y_samples = [np.asarray([[word_to_index[w] for w in tweet[1:]] for tweet in itweets]) for itweets in tweets]

# randomly permute the samples
for length_index in range(0, len(X_samples)):
    p = np.random.permutation(len(X_samples[length_index]))
    X_samples[length_index] = X_samples[length_index][p]
    Y_samples[length_index] = Y_samples[length_index][p]   

#save training data
print ("Saving training data to %s." % training_file)
np.savez(training_file, X_samples=X_samples, Y_samples=Y_samples)
print ("Saving additional tweet data to %s." % data_file)
dfile = open(data_file, "wb")
pickle.dump((usernames, numbers, links), dfile)