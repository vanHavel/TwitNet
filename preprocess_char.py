import numpy as np
import os
import sys
import argparse

import utility.process

# constants
newline = "<newline>"

# parse command line arguments
parser = argparse.ArgumentParser(description="Preprocess tweet data for character based language model.")
parser.add_argument("-i", "--input_file", default="data/tweets.txt", help="Path to tweet input file.")
parser.add_argument("-l", "--history_length", default=40, type=int, help="Maximum length of char history used for backpropagation through time steps.")
args = parser.parse_args()

input_file = args.input_file
history = args.history_length

# check arguments
if history < 2:
    print ("Invalid history value, must be greater than 1.")
    sys.exit(1)

# create paths to output files
(drive, filename) = os.path.split(input_file)
vocab_file = os.path.join(drive, "vocab.txt")
training_file = os.path.join(drive, "training_data.npz")

# open input file
print ("Reading input tweets.")
ifile = open(input_file, "r")
lines = ifile.readlines()
ifile.close()
        
# count word occurrences
print ("Building vocabulary.")
vocab = []
for line in lines:
    for char in line:
        if char not in vocab:
            vocab.append(char)

# build index to char, char to index tables, handling newline character
index_to_char = [(c if c != "\n" else newline) for c in vocab]
char_to_index = dict([(c,i) for i,c in enumerate(index_to_char)])

# write vocab to file
print ("Writing vocabulary to %s." % vocab_file)
ofile = open(vocab_file, "w")
for c in index_to_char:
    ofile.write(c + "\n")
ofile.close()

# Create the training data
print ("Creating training data.")
# concat all lines
corpus = ""
for line in lines:
    corpus += line
    
# map to indices, handling newline character
corpus = [(char_to_index[c] if c != "\n" else char_to_index[newline]) for c in corpus]

# for efficiency: use bytes to store small numbers
if len(index_to_char) < 256:
    int_size = np.uint8
else:
    int_size = np.uint32

# split into samples of length history
num_samples = int(np.floor(len(corpus) / history))
samples = [corpus[i * history : i * history + history] for i in range(0, num_samples)]
X_samples = np.array([sample[:-1] for sample in samples], dtype=int_size)
Y_samples = np.array([sample[1:] for sample in samples], dtype=int_size)

# randomly permute the samples
p = np.random.permutation(len(X_samples))
X_samples = X_samples[p]
Y_samples = Y_samples[p]   

#save training data
print ("Saving training data to %s." % training_file)
np.savez(training_file, X_samples=[X_samples], Y_samples=[Y_samples])