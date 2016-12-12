# TwitNet
Some scripts for training a neural network language model on tweets.

## Introduction
TwitNet is a colection of python scripts designed that allow to easily create RNN language models to train on tweet data and sample tweets from. The scripts handle basic preprocessing, creating models of different size and architecture, training these models and sampling from them. TwitNet is built on top of the Keras library for neural networks in python, which in itself uses either TensorFlow or Theano as backend for tensor operations.

TwitNet was mainly developed with word based language models in mind, where the preprocessing takes care of replacing links and twitter usernames with special tokens so that these words don't clutter up the vocabulary. But there is also support for character based language models with more basic preprocessing.

## Quick Start Guide
* Install the necessary dependencies: Numpy, TensorFlow and Keras. See Installation for details.
* Gather a corpus of tweets and store them as data/tweets.txt in the format of one tweet per line.
* To create and train a word based language model with the default parameters (vocabulary size of 4000 and 2 layer LSTM with 128 units each), run
 
 ```python
 python preprocess.py
 python create_model.py
 python train_model.py
 ```
 
 This will train the model for 50 epochs, print the loss on 10% of the training data as validation data and save the parameters after every epoch. 
* You can then sample tweets from the model with the optimal loss by running

 ```python
 python sample.py -m model/model_epoch<n>.mod
 ```
 
 where \<n\> is the number of the epoch after which the loss was minimal.
* To train a character based model, run `python preprocess_char.py` instead of `python preprocess.py` and `python sample_char.py` instead of `python sample.py`.

## Installation
1. Install [Python 3](https://www.python.org).
2. Install [Numpy](http://www.numpy.org) (`pip install numpy`).
3. Install [TensorFlow](https://www.tensorflow.org) (`pip install tensorflow`).
4. Install [Keras](https://keras.io) (`pip install keras`).
5. (Optional, but recommended!) Install [Theano](http://deeplearning.net/software/theano/) (`pip install theano`) and [setup Keras to use Theano as backend](https://keras.io/#switching-from-tensorflow-to-theano). This will greatly decrease training time as Theano has better performance training recurrent neural networks.
6. Clone this repository. You should now be able to run the scripts.

## Detailed description of scripts
### preprocess.py
Preproccesses a file of tweets, given in a text file in the format of one tweet per line. This splits up the tweets into words and (if not turned off by the flag) performs some more preprocessing steps: 

* Every link is replaced by the \<link\> token, every number by the \<number\> token, and every twitter username by the \<user\> token. Hashtags are not replaced. 

 The idea behind this is that hashtags might add some significant meaning to the tweets, while concrete links, numbers or usernames are less important. This is of course debatable, and this preprocessing step can be turned off - in this case it might be necessary to also increase the vocabulary size.
 
 The replaced links, names and numbers are stored in an additional tweet data file in the same directory as the input tweet file. Tokens appearing multiple times will also be stored with their multiplicity. During sampling, the tokens are replaced by samples chosen uniformly at random from this stored data.
* The vocabulary is limited to the most frequent words, where the vocabulary size is given as command line argument (default: 4000). Every word not in the vocabulary is replaced by the \<unknown\> token.
* Very short tweets with less than the given minimum length are removed.
* Words are mapped to indices and the tweets are stored as training sequences for the language model. The training data is stored in the same directory as the tweet input data.

#### Command Line Arguments
| Short name | Long name | Argument type | Default value | Description |
|---|---|---|---|---|
| `-i` | `--input_file` | String | "data/tweets.txt" | Path to tweet input file. |
| `-v` | `--vocab_size` | Integer | 4000 | Size of the vocabulary. |
| `-m` | `--min_length` | Integer | 3 |  Minimum word length of a tweet. |
| `-c` | `--case_sensitive` | Flag | False | If set, handle words case-sensitive. |
| `-u` | `--tokens_unchanged` | Flag | False | If set, do not replace individual links, usernames and numbers. |
| `-h` | `--help` | Flag | False | Print help text. |
### preprocess_char.py
Preprocesses a file of tweets in the same input format as for `preprocess.py`, but training data is created for training a character based language model. For this, the whole tweet corpus is simply concatenated and split into fixed length character sequences. No replacement of links etc. is performed and the vocabulary is not limited, since it will typically be small(\< 100). 

An important parameter is the history length, which determines the length of training sequences and thus the maximum number of backpropagation through time steps. The default history length is 40 characters.
#### Command Line Arguments
TODO
### create_model.py
Creates a keras recurrent neural network model with the parameters given on the command line. The structure of the network is

* One initial embedding layer from the vocabulary size to the hidden size. This layer is omitted if the vocabulary size is not at least five times as large as the hidden size, which typically happens in character based models.
* A specified number of recurrent layers(default: 2) with a specified number of hidden units per layer(default: 128). The default architecture is LSTM, but RNNs and GRUs are also available.
* Optinally dropout is performed after each recurrent layer with given retention probability.
* A final dense layer from the hidden size to the vocabulary size.

As optimizer either Adam or RMSProp can be chosen and the initial learning rate can be specified. The model is compiled and stored at the specified output location.
#### Command Line Arguments
TODO
### train_model.py
Trains a given model for a specified number of epochs. For this, the training data is split into training and validation data, and the loss is evaluated regularly on the validation data. The model parameters are also saved regularly. If desired, the model can be trained on the complete training data, with the potential danger of overfitting - the loss will then be evaluated on a part of the training data. The learning rate is adjusted automatically once the loss stops decreasing.
#### Command Line Arguments
TODO
### sample.py
Samples tweets from a trained word based language model. For this purpose the user can supply an initial sequence to the model which is then completed into a number of tweets of specified maximum length. 

If special tokens for links etc. where created during preprocessing, they are replaced with values sampled uniformly at random from the stored tweet data. Unless specified by the corresponding flag, no \<unknown\> tokens will be sampled. One can also experiment with a temperature argument for sampling, where a temperature < 1 will lead to a less random output. Note that it will initially take several seconds to load the model parameters.
#### Command Line Arguments
TODO
### sample_char.py
Like `sample.char`, except that tweets are sampled from a character based language model. 
#### Command Line Arguments
TODO
