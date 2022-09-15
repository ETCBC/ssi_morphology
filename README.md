# A Parser for Biblical Hebrew and Syriac Morphology
This repository contains code and data for the OpenSSI Hebrew/Syriac morphology project of the [Netherlands eScience Center](https://www.esciencecenter.nl/) and the [Eep Talstra Centre for Bible and Computer (ETCBC)](www.etcbc.nl)) of the [Faculty of Religion and Theology, Vrije Universiteit](https://frt.vu.nl/nl/index.aspx) entitled “Morphological Parser for Inflectional Languages Using Deep Learning”. The project is based on the experience that has been built up in more than four decades of the computational linguistic analysis of the Hebrew Bible at the Eep Talstra Centre for Bible and Computer (ETCBC) and some of the unique aspects of the encoding of the ETCBC linguistic database of the Hebrew Bible. These unique aspects do justice to the fact that Biblical Hebrew, like many other Semitic languages, is an inflectional language with a rich morphology. What can be said with one word in Biblical Hebrew sometimes needs five or six words in an English translation. Therefore, to add linguistic annotations to a text, it is better to encode the smaller parts of a word (morphemes) rather than the complete words (as is usually done in, e.g., the preparation of English or Dutch text corpora). This, however, is very labour-intensive. The new project will endeavour to use Machine Learning to automate this process for Hebrew and Syriac texts.

# Dependencies
The dependencies are Pytorch, Scikit Learn, Tensorvoard and python-Levenshtein (see also requirements.txt).
It works best if you use a GPU.

# Data
In the folder "data" some datasets are provided. All datasets are tab separated files, containing 4 columns: book, chapter, verse, text.
The text is in ETCBC transcription.

- t-in_con: raw unvocalized text of the MT.
- t-in_voc: raw vocalized text of the MT.
- t-out: morphologically parsed MT.
- s-in: raw Syriac text.
- s-out: morphologically parsed Syriac text.

Models are saved in the folder transformer_models.
The results of evaluation on the test set are saved in the folder evaluation_results_transformer.

# Usage
There are three different ways you can train a model:

### 1. Train on a single dataset
You can train a model which can parse Hebrew or Syriac morphology. A minimal input would be:

Train model on vocalized Hebrew input data:
- python main.py -i=t-in_voc -o=t-out -ep=2 -l=5 -lr=0.0001

Train model on unvocalized data:
- python main.py -i=t-in_con -o=t-out -ep=2 -l=5 -lr=0.0001

Train model on Syriac data:
- python main.py -i=s-in -o=s-out -ep=2 -l=5 -lr=0.0001

The required command line arguments are:
- i Filename of file with input sequences.
- o Filename of file with output sequences.
- l Number of graphical units in input sequence, e.g, "BR>CJT" has length 1, "BR>CJT BR>" has length 2, etc.
- ep Number of training epochs.
- lr Learning rate.

### 2. Train on two datasets (data are mixed)
It is also possible to train a model on Hebrew and Syriac data together. Here, data are imported and mixed before training takes place.
- python main.py -i=t-in_voc -o=t-out -i2=s-in -o2=s-out -ep=2 -l=5 -lr=0.0001

There are two extra arguments:
- i2 the second input file.
- o2 the second output file.

### 3. Train on two datasets (sequentially)
You can also apply Transfer Learning. With Transfer Learning, you will first train the model on Hebrew data, after which you continue training the model on Syriac data.
- python main.py -i=t-in_voc -o=t-out -i2=s-in -o2=s-out -ep=2 -ep2=2 -l=5 -lr=0.0001

There is an extra command argument:

- ep2 Number of epochs for training the second (Syriac) dataset. With this argument the script will recognize that sequential training is needed.

An optional argument is:

- et Evaluate on test set. Default is False, if set to True, the model will be evaluated on the test set. In the case of training on two datasets, the test set will contain data from the second output file (o2).

There is a number of other optional arguments, which can be used to optimize the model:

- emb Embedding size, must be divisible by number of heads, default=512.
- nh Number of heads, default=8.
- nel Number of layers in encoder", default=3
- ffn Feed Forward Network Hidden Dimension, default=512.
- ndl Number of layers in decoder, default=3.
- dr Dropout in transformer model, default=0.1.
- b Batch size during training, default=128.
- wd Weight decay, default=0.0.

![OpenSSI2021_ETCBC](https://user-images.githubusercontent.com/7325578/118670815-3b9ecc80-b7f7-11eb-9beb-cf992c830039.jpg)
