### Recurrent & Convolutional Neural Network Modules

This repo contains Theano implementations of popular neural network components and optimization methods. Source code of the following papers are also available:

[1] [Semi-supervised Question Retrieval with Gated Convolutions](http://arxiv.org/abs/1512.05726). Lei et. al. NAACL 2016

[2] [Molding CNNs for text: non-linear, non-consecutive convolutions](http://arxiv.org/abs/1508.04112). Lei et. al. EMNLP 2015

#### Features
  - Basic modules including feedforward layer, dropout, word embedding, RNN, LSTM, GRU and CNN
  - Optimization methods including SGD, AdaGrad, AdaDelta and Adam
  - Advanced modules from recent papers such as attention and gated convolution.
  - Transparent to use GPU

#### Projects

##### 1. Neural Question Retrieval for Community-based QA

The directories [code/qa](/code/qa) and [code/pt](/code/pt) contain the implementation of the model described in paper [[1]]((http://arxiv.org/abs/1512.05726)).

Datasets and and pre-trained word vectors are available at [this repo](https://github.com/taolei87/askubuntu).

<br>

##### 2. Sentiment Analysis / Document Classification

The directory [code/sentiment](/code/sentiment) contains the implementation of the model described in paper [[2]](http://arxiv.org/abs/1508.04112).

Datasets and and pre-trained word vectors are available at [this repo](https://github.com/taolei87/text_convnet).

<br>

#### Dependencies
  [Theano](http://deeplearning.net/software/theano/) >= 0.7, Python >= 2.7, Numpy

  
