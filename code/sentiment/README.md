#### Sentiment analysis (and other sequence classification task)

This directory contains the re-factored implementation of the non-consecutive CNN model (StrCNN), originally at [[1]](https://github.com/taolei87/text_convnet). The code is now more modular, and I add an option to choose layer type among StrCNN, its adaptive version [RCNN](http://arxiv.org/pdf/1512.05726v2.pdf) or LSTM.

##### Data
The data for sentiment analysis can be downloaded at the [orignal repo](https://github.com/taolei87/text_convnet).


##### Usage
  1. Clone the rcnn repo, and switch to “dev.0” branch
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
