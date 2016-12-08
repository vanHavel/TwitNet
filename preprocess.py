import numpy as np
import operator
import os
import sys
import argparse
import pickle

import utility_process

tweet_start = "<tweet_start>"
tweet_end = "<tweet_end>"
unknown_token = "<unknown>"
link_token = "<link>"
number_token = "<number>"
user_token = "<username>"

# parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess tweet data.")
parser.add_argument("-v", "--vocab_size", default=5000, type=int, help="Size of the vocabulary.")
parser.add_argument("-m", "--min_length", default=4, type=int, help="Minimum word length of a tweet.")
parser.add_argument("-c", "--case_sensitive", action="store_true", help="Handle words case-sensitive.")
parser.add_argument("-u", "--tokens_unchanged", action="store_true", help="Do not replace individual links, usernames etc.")
parser.add_argument("-i", "--input_file", default="data/tweets.txt")
args = parser.parse_args()

input_file = args.input_file
vocab_size = args.vocab_size
min_length = args.min_length
case_sensitive = args.case_sensitive
links_unchanged = args.links_unchanged

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
(usernames, numbers, links, lines) = utility.tokenize(lines, unchanged=tokens_unchanged)
    
# map words to lowercase if flag is set
if not case_sensitive:
    for i, line in enumerate(lines):
        lines[i] = [word.lower() for word in line]
         
# replace links if flag is set
if not links_unchanged:
    for i, line in enumerate(lines):
        lines[i] = [(link_token if (len(word) > 2 and word[0:2] == "//") else word) for word in line]
        
# count word occurrences
print ("Building vocabulary.")
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

#save training data
print ("Saving training data to %s." % training_file)
np.savez(training_file, X_samples=X_samples, Y_samples=Y_samples)
print ("Saving additional tweet data to %s." % data_file)
dfile = open(data_file, "wb")
pickle.dump((usernames, numbers, links), dfile)