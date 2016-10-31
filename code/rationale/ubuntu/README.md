
Code for AskUbuntu data. 

[/pre-trained](pre-trained) contains a model by pre-training encoder using the [code](../../pt/), following our [previous work](https://arxiv.org/pdf/1512.05726.pdf) on semi-supervised question retrieval.

The data and pre-trained word embeddings are available at [this repo](https://github.com/taolei87/askubuntu).

#### Code Usage

To run the code, you need Numpy and Theano (> 0.7.0.dev-8d3a67 I used) installed. Next:
  1. Clone the rcnn repo
  2. Use “export PYTHONPATH=/path/to/rcnn/code” to add the rcnn/code directory to Python library
  3. Run `python rationale.py --help` to see all running options

Example run of AskUbuntu data:
```
THEANO_FLAGS='device=gpu,floatX=float32'        # use GPU and 32-bit float
python rationale.py                             # independent selection version
      --embeddings /path/to/vectors             # path to load word vectors (required)
      --corpus text_tokenized.txt.gz            # path to the AskUbuntu corpus (required)
      --train train_random.txt                  # path to training set (required)
      --dev dev.txt                             # path to development set (required)        
      --test test.txt                           # path to test set (required)
      --load_pretrain pre-trained/rcnn_lr0.001_d400_dr0.1.pkl.gz
                                                # path to pre-trained encoder (required)
      --dump_rationale outputs.txt              # dump selected rationales
      --sparsity 0.01 --coherent 1.0            # regularizations
      --joint 1                                 # update both encoder and generator (0: generator only)
      --merge 1                                 # put question title at the beginning (1) or randomly (2)
```
