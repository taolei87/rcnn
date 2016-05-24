#### Sentiment analysis (and other sequence classification task)

This directory contains the re-factored implementation of the non-consecutive CNN model (StrCNN), originally at [[1]](https://github.com/taolei87/text_convnet). The code is now more modular, and I add an option to choose layer type among StrCNN, its adaptive version [RCNN](http://arxiv.org/pdf/1512.05726v2.pdf) or LSTM.

##### Data
The data for sentiment analysis can be downloaded at the [orignal repo](https://github.com/taolei87/text_convnet). 

To apply the model on other task(s), you can create the training / evaluation data in the following format:
```
4 béart and berling are both superb , while huppert ... is magnificent .
1 apparently reassembled from the cutting-room floor of any given daytime soap .
2 the entire movie is filled with deja vu moments .
0 final verdict : you 've seen it all before .
3 may be more genial than ingenious , but it gets the job done .
```
Each line represents an input instance that starts with the label ID, followed by the list of words in the sentence/document. The label ID has to be integer and the smallest ID should be 0. The ID and words are separated by whitespaces.

##### Usage
  1. Clone the rcnn repo
  2. Use “export PYTHONPATH=/path/to/rcnn/code” to add the rcnn/code directory to Python library
  3. Run `python main.py --help` to see all running options

Some examples:
```
python model.py --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --train data/stsa.binary.phrases.train  \
    --dev data/stsa.binary.dev  --test data/stsa.binary.test  \
    --save output_model
```

Use `--load` to load a saved model and re-test it on a test set:
```
python model.py --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --test data/stsa.binary.test  \
    --load output_model
```

We can optionally specify Theano configs via `THEANO_FLAGS`:
```
THEANO_FLAGS='device=gpu,floatX=float32'; python model.py ...
```

Examples with more hyperparamter settings:
```
THEANO_FLAGS='device=gpu,floatX=float32'; python model.py  \
    --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --train data/stsa.binary.phrases.train  \
    --dev data/stsa.binary.dev  --test data/stsa.binary.test  \
    --save output_model  \
    --layer rcnn \
    --depth 2  --order 2 --hidden_dim 250  \
    --dropout_rate 0.3 --act relu  \
    --learning adam  --learning_rate 0.001
```

```
THEANO_FLAGS='device=gpu,floatX=float32'; python model.py  \
    --embedding word_vectors/stsa.glove.840B.d300.txt.gz  \
    --train data/stsa.binary.phrases.train  \
    --dev data/stsa.binary.dev  --test data/stsa.binary.test  \
    --save output_model  \
    --layer strcnn
    --depth 3  --order 3  --decay 0.5  --hidden_dim 200  \
    --dropout_rate 0.3  --l2_reg 0.00001  --act relu  \
    --learning adagrad  --learning_rate 0.01
```

##### Hyperparameters

To optimize the model performance, you can tune the following options:

 1. Layer type (--layer):  strcnn, rcnn, lstm
 2. Activation (--act): relu, tanh, etc (lstm always uses tanh)
 3. Poolling (--pooling): 0 or 1  (whether use mean pooling or just take the last state)
 4. Number of layers (--depth): 1,2,3 etc
 5. Dropout (--dropout), L2 regularization (--l2_reg) and hidden dimension (-d)
 6. Learning method (--learning): adam, adagrad, adadelta etc.
 7. Learning rate (--learning_rate): 0.001, 0.01 etc
 8. Feature filter width (--order): 2, 3, etc (only valid for strcnn and rcnn)
 9. Non-consecutive decay (--decay): 0.3, 0.5 etc (only valid for strcnn; rcnn uses adaptive decay)
