### Recurrent & Convolutional Neural Network Modules

This repo contains Theano implementations of popular neural network components, optimization methods and new architectures used in recent research papers.

Source code of NN library at [code/nn/](/code/nn)

#### Features
  - Basic modules including feedforward layer, dropout layer, word embedding layer, RNN, LSTM, GRU and CNN
  - Optimization methods including SGD (with gradient clipping), AdaGrad, AdaDelta and Adam
  - Advanced modules from recent papers:
    - non-consecutive, non-linear CNN (http://arxiv.org/abs/1508.04112)
    - attention layer, non-recurrent version (http://arxiv.org/abs/1509.06664)
    - attention layer, non-recurrent bi-linear version (http://arxiv.org/pdf/1509.00685.pdf)
    - recurrent convolutional network (http://arxiv.org/abs/1512.05726)
  - Transparent to use GPU

#### Dependencies
  [Theano](http://deeplearning.net/software/theano/) >= 0.7, Python >= 2.7, Numpy
  
<br>

----------

### Projects
#### 1. Similar Question Retrieval 

The directories [code/qa](/code/qa) and [code/pt](/code/pt) contain the implementation of the model described in the paper [Semi-supervised Question Retrieval with Gated Convolutions](http://arxiv.org/abs/1512.05726).

Data and pre-trained word vectors are available at [this repo](https://github.com/taolei87/askubuntu).
