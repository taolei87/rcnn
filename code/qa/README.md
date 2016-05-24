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

Here is an example to run the model with default parameters:
```
THEANO_FLAGS='device=gpu,floatX=float32'        # use GPU and 32-bit float
python main.py --corpus path/to/corpus          # path to the corpus file
      --train path/to/train                     # path to training file
      --dev path/to/dev        
      --test path/to/test      
      --dropout 0.1                             # dropout probability
      -d 400                                    # hidden dimension
      --save_model model.pkl.gz                 # save trained model to this file
```
