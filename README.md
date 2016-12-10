# TwitNet
Some scripts for training a neural network language model on tweets.

## Introduction
TwitNet is a colection of python scripts designed that allow to easily create RNN language models to train on tweet data and sample tweets from. The scripts handle basic preprocessing, creating models of different size and architecture, training these models and sampling from them. TwitNet is built on top of the Keras library for neural networks in python, whhich in itself used either TensorFlow or Theano as backend for tensor operations.
TwitNet was mainly developed with word based language models in mind, where the preprocessing takes care of replacing links and twitter usernames with special tokens so that these words don't clutter up the vocabulary. But there is also support for character based language models with more basic preprocessing.

## Quick Start Guide
Install the necessary dependencies: Numpy, TensorFlow and Keras. See Installation for details.
Gather a corpus of tweets and store them as data/tweets.txt in the format of one tweet per line.
To create and train a word based language model with the default parameters (vocabulary size of 4000 and 2 layer LSTM with 128 units each), run
python preprocess.py
python create_model.py
python train_model.py
This will train the model for 50 epochs, print the loss on 10% of the training data as validation data and save the parameters after every epoch. 
You can then sample tweets from the model with the optimal loss by running
sample.py -m model/model_epoch<n>.mod
where <n> is the number of the epoch after which the loss was minimal.
To train a character based model, run preprocess_char.py instead of preprocess.py and sample_char.py instead of sample.py.

## Installation
0. Install Python 3 and pip.
1. Install Numpy (`pip install numpy`).
2. Install TensorFlow (`pip install tensorflow`).
3. Install Keras (`pip install keras`).
4. (Optional, but recommended!) Install Theano (`pip install theano) and setup Keras to use Theano as backend. This will greatly decrease training time as Theano has better performance training recurrent neural networks.
5. Clone this repository. You should now be able to run the scripts.

## Detailed description of scripts
TODO
