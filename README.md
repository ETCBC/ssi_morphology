# A Parser for Biblical Hebrew and Syriac Morphology
This repository contains code and data for the OpenSSI Hebrew/Syriac morphology project of the [Netherlands eScience Center](https://www.esciencecenter.nl/) and the [Eep Talstra Centre for Bible and Computer (ETCBC)](www.etcbc.nl)) of the [Faculty of Religion and Theology, Vrije Universiteit](https://frt.vu.nl/nl/index.aspx) entitled “Morphological Parser for Inflectional Languages Using Deep Learning”. The project is based on the experience that has been built up in more than four decades of the computational linguistic analysis of the Hebrew Bible at the Eep Talstra Centre for Bible and Computer (ETCBC) and some of the unique aspects of the encoding of the ETCBC linguistic database of the Hebrew Bible. These unique aspects do justice to the fact that Biblical Hebrew, like many other Semitic languages, is an inflectional language with a rich morphology. What can be said with one word in Biblical Hebrew sometimes needs five or six words in an English translation. Therefore, to add linguistic annotations to a text, it is better to encode the smaller parts of a word (morphemes) rather than the complete words (as is usually done in, e.g., the preparation of English or Dutch text corpora). This, however, is very labour-intensive. The new project will endeavour to use Machine Learning to automate this process for Hebrew and Syriac texts.

# Dependencies
You install most dependencies with:

`pip install -r requirements.txt`

For the installation of Pytorch, see [the Pytorch website](https://pytorch.org/).

# Data
In the folder "data" some datasets are provided. These datasets are tab separated files, containing 4 columns: book, chapter, verse, text.
The text is in ETCBC transcription.

- t-in_con: raw unvocalized text of the MT.
- t-in_voc: raw vocalized text of the MT.
- t-out: morphologically parsed MT.
- s-in: raw Syriac text.
- s-out: morphologically parsed Syriac text.

Models are saved in the folder transformer_models.
The results of evaluation on the test set are saved in the folder evaluation_results_transformer.

# Train a Model
There are three different ways you can train a model:

### 1. Train on a single dataset
You can train a model which can parse Hebrew or Syriac morphology. It works best if you use a GPU. A minimal input would be:

Train model on vocalized Hebrew input data:
- `python main.py -mo=train -i=t-in_voc -o=t-out -ep=2 -l=5 -lr=0.0001`

Train model on unvocalized data:
- `python main.py -mo=train -i=t-in_con -o=t-out -ep=2 -l=5 -lr=0.0001`

Train model on Syriac data:
- `python main.py -mo=train -i=s-in -o=s-out -ep=2 -l=5 -lr=0.0001`

The required command line arguments are:
- mo Mode, can be train or predict.
- i Filename of file with input sequences.
- o Filename of file with output sequences.
- l Number of graphical units in input sequence, e.g, "BR>CJT" has length 1, "BR>CJT BR>" has length 2, etc.
- ep Number of training epochs.
- lr Learning rate.

### 2. Train on two datasets (data are mixed)
It is also possible to train a model on Hebrew and Syriac data together. Here, data are imported and mixed before training takes place.
- `python main.py -mo=train -i=t-in_voc -o=t-out -i2=s-in -o2=s-out -ep=2 -l=5 -lr=0.0001`

There are two extra arguments:
- i2 the second input file.
- o2 the second output file.

### 3. Train on two datasets (sequentially)
You can also apply Transfer Learning. With Transfer Learning, you will first train the model on Hebrew data, after which you continue training the model on Syriac data.
- `python main.py -mo=train -i=t-in_voc -o=t-out -i2=s-in -o2=s-out -ep=2 -ep2=2 -l=5 -lr=0.0001`

There is an extra command argument:

- ep2 Number of epochs for training the second (Syriac) dataset. With this argument the script will recognize that sequential training is needed.

Optional arguments are:

- et Evaluate on test set. Default is False, if set to True, the model will be evaluated on the test set. In the case of training on two datasets, the test set will contain data from the second output file (o2).
- sz This is the beam size during beam search decoding. Default is 3. If one chooses 0, a greedy decoder is used. In general, beam search gives better accuracy, but is slower than the greedy decoder.
- ba Is the beam alpha value. Default is 0.75. This regulates the penalty for longer sequences. See also [this paper](https://arxiv.org/pdf/1609.08144.pdf), page 12. 

There is a number of other optional arguments, which can be used to optimize the model:

- emb Embedding size, must be divisible by number of heads, default=512.
- nh Number of heads, default=8.
- nel Number of layers in encoder, default=3
- ffn Feed Forward Network Hidden Dimension, default=512.
- ndl Number of layers in decoder, default=3.
- dr Dropout in transformer model, default=0.1.
- b Batch size during training, default=128.
- wd Weight decay, default=0.0.

# Make predictions on new data

You can make predictions using a trained model on new data, for which you need the following things:
1. A file with new data in the folder new_data.
2. A trained model.
3. A configuration file of the model.
4. A YAML file, which should be in the folder new_data, in which you indicate where the new data, the model, the model configuration file, and the name of the output file can be found.

- ad 1. the data file is tab separated and should consist of four columns, without a header. The columns are book, chapter, verse, and text.
- ad 2. The model should be in a subfolder in the folder transformer models.
- ad 3. The configuration file of the model is in the same folder as the model. The configuration file contains the hyperparameters needed to initialize the model. 
      Next to that, it contains the sequence length used to slice the text, and 2 dictionaries that are used to convert the input text to integers, and integers to output text.
- ad 4. The name of the output file is optional in the YAML. If no output filename is given, the results are written to standard output. You can also indicate the index in the sequence for which you want the prediction (If this is not indicated, the predict_idx is set to 0.), the beam size (0 for greedy decoding), and the beam alpha value. The YAML file has the following structure:

```
model_info:
    folder: name_of_folder_containing_model_and_config
    model_config: model_config_file_name
    model: model_name
new_data: new_data_file_name
output: output_file_name
predict_idx: idx
beam_size: beam_size
beam_alpha: beam_alpha
```

You can run a prediction with:

`python main.py -mo=predict -pcf=yaml_file_name`

![OpenSSI2021_ETCBC](https://user-images.githubusercontent.com/7325578/118670815-3b9ecc80-b7f7-11eb-9beb-cf992c830039.jpg)
