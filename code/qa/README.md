#### Similar Question Retrieval

This directory contains the implementation of the neural method described in the [paper](https://arxiv.org/pdf/1512.05726.pdf). The method is used to calculate text similarity, for applications such as similar question retrieval in community-based QA forums.

##### Data
The data used in this work is taken from the [AskUbuntu](http://askubuntu.com/) 2014 dump.  The processed data can be downloaded at this [repo](https://github.com/taolei87/askubuntu).

##### Dependencies
To run the code, you need the following extra packages installed:
  - [PrettyTable](https://pypi.python.org/pypi/PrettyTable) (only for this project)
  - [Scikit-Learn](http://scikit-learn.org/stable/) (only for this project)
  - Numpy and Theano (required in general for this repository)


##### Usage
  1. Clone the rcnn repo
  2. Use “export PYTHONPATH=/path/to/rcnn/code” to add the rcnn/code directory to Python library
  3. Run `python main.py --help` to see all running options

To specify Theano configs, run the code via ```THEANO_FLAGS='...' python main.py ...``` For instance, here is an example to run the model with default parameters:
```
THEANO_FLAGS='device=gpu,floatX=float32'        # use GPU and 32-bit float
python main.py --corpus path/to/corpus          # path to the corpus file
      --embeddings /path/to/vectors             # path to load word vectors
      --train path/to/train                     # path to training file
      --dev path/to/dev        
      --test path/to/test      
      --dropout 0.1                             # dropout probability
      -d 400                                    # hidden dimension
      --save_model model.pkl.gz                 # save trained model to this file
```
The corpus, training/development/test files and the word vectors are available at the [data repo](https://github.com/taolei87/askubuntu). 

The above example trains a model from scratch.
To fine tune a model that is pre-trained using unlabeled text (see [code/pt](../pt) directory for more information), use the ```--load_pretrain``` option:
```
THEANO_FLAGS='device=gpu,floatX=float32'        
python main.py --corpus path/to/corpus
      --embeddings /path/to/vectors
      --train path/to/train                    
      --dev path/to/dev        
      --test path/to/test      
      --dropout 0.1                 
      -d 400                          
      --save_model model.pkl.gz
      --load_pretrain path/to/pretrained/model
```

##### Other hyperparameters
You can train the model with different settings by specifying the following options:
 1. Layer type (--layer):  rcnn, lstm, gru
 2. Activation (--act): relu, tanh, etc 
 3. Average pooling (--average): 0 or 1  (whether use mean pooling or just take the last state)
 4. Number of layers (--depth)
 5. Dropout (--dropout), L2 regularization (--l2_reg) and hidden dimension (-d)
 6. Learning method (--learning): adam, adagrad, adadelta etc.
 7. Learning rate (--learning_rate): 0.001, 0.01 etc
 8. Feature filter width (--order): 2, 3, etc
