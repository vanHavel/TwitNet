import argparse
import sys
import random
import string

import utility.model

from keras.models import load_model

# constants
newline = "<newline>"

# parse command line arguments
parser = argparse.ArgumentParser(description="Sample tweets from a model.")
parser.add_argument("-m", "--model_file", default="model/model.mod", help="Path to model file.")
parser.add_argument("-v", "--vocab_file", default="data/vocab.txt", help="Path to vocabulary file.")
parser.add_argument("-n", "--samples_number", default=3, type=int, help="Number of samples to create for each user input.")
parser.add_argument("-l", "--max_length", default=140, type=int, help="Maximum number of characters in a tweet.")
parser.add_argument("-t", "--temperature", default=1.0, type=float, help="Temperature for sampling from the network's output distribution.")
args = parser.parse_args()

model_file = args.model_file
vocab_file = args.vocab_file
samples_number = args.samples_number
max_length = args.max_length
temperature = args.temperature

# check command line arguments
if temperature < 0.0:
    print ("Invalid temperature value, may not be negative.")
    sys.exit(1)
if samples_number <= 0:
    print ("Invalid number of samples, must be positive.")
    sys.exit(1)
if max_length < 1: 
    print ("Invalid max length, must be greater than 1.")
    sys.exit(1)

# read vocabulary
print ("Reading vocab.")
vfile = open(vocab_file, "r")
index_to_char = [line[:-1] for line in vfile.readlines()]
vocab_size = len(index_to_char)
char_to_index = dict([(c,i) for i,c in enumerate(index_to_char)])
vfile.close()

# load model
print ("Loading model.")
model = load_model(model_file)

# interactive loop
print ("Enter an initial sentence fragment for the model.") 
print ("Alternatively hit enter for a random start to the tweet.")
print ("Enter :q to exit.")
user_input = input()
while user_input != ":q":
    if user_input != "":
        # use the user input
        seq = list(user_input)
    else:
        # start with random letter
        seq = [random.choice(string.ascii_letters)]
    # filter out unknown chars and transform to index 
    seq_filtered = filter(lambda c: c in char_to_index, seq)
    seq_indexed = [char_to_index[c] for c in list(seq_filtered)]
    for i in range(0, samples_number):
        generated = utility.model.generate_tweet(model, seq_indexed, char_to_index[newline], vocab_size,
                                                 max_length=max_length, temperature=temperature, sample_unknown=True)
        generated_chars = [index_to_char[i] for i in generated]
        output = "".join(generated_chars)
        output = output.replace(newline, "\n")
        print(output)
    user_input = input()
        