import nltk
import argparse
import sys

import utility

from keras.models import load_model

# constants
tweet_start = "<tweet_start>"
tweet_end = "<tweet_end>"
unknown_token = "<unknown>"
link_token = "<link>"

# parse command line arguments
parser = argparse.ArgumentParser(description="Sample tweets from a model.")
parser.add_argument("-m", "--model_file", default="model/model.mod", help="Path to model file.")
parser.add_argument("-v", "--vocab_file", default="data/vocab.txt", help="Path to vocabulary file.")
parser.add_argument("-n", "--samples_number", default=3, type=int, help="Number of samples to create for each user input.")
parser.add_argument("-l", "--max_length", default=32, type=int, help="Maximum number of words in a tweet.")
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
index_to_word = [line[:-1] for line in vfile.readlines()]
vocabsize = len(index_to_word)
word_to_index = dict([(c,i) for i,c in enumerate(index_to_word)])
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
        seq = [tweet_start] + nltk.word_tokenize(user_input)
    else:
        seq = [tweet_start]
    seq_indexed = [(word_to_index[w] if w in word_to_index else word_to_index[unknown_token]) for w in seq]
    for i in range(0, samples_number):
        generated = utility.generate_tweet(model, seq_indexed, max_length, temperature, word_to_index[tweet_end])
        generated_words = [index_to_word[i] for i in generated]
        generated_words = utility.postprocess(generated_words)
        print(generated_words)
    user_input = input()
        