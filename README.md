# Translator cs -> en

This is a project completed as a term project for course **PV061 Machine Translation** at FI MUNI. The goal is to fine-tune a pretrained neural translator on a different dataset.

This document describes:
1) source data
1) general info about repository
1) preparation dataset
1) training of the model
1) using model for inference
1) evaluation of the model
1) how to use the project on Apollo server


## Source data

The dataset used in the project is the **CZENG 2.0** released by Institute of Formal and Applied Linguistics (ÚFAL) at Charles University.
Czeng contains roughly 700 million paralel sentences (and even more synthetically translated from monolingual intended for backtranslation training).

The data in Czeng comes from various sources. One of them is a webscrape, which means that it contains some auto-translated sentences or even untranslated ones (sentence in both languages being exactly the same string).

Fortunately, each paralel sentence in Czeng is scored by a neural translator from Charles university which aims to ranks the legitimity of the translation. Each sentence is also scored using a score computed from pseudo-probabilities from language models (one for source and one for target language). This can be used as a filter on the data.


## Code

The repository contains one package called `translator` which provides a command line program which can be invoked using command `translator`.

All code is in `src/translator/`. It contains a `main.py` which provides cli interface for all functionality - preparation of dataset, training, interactive inference, translation of text files and evaluation.

To set up the project, you will need Python 3.8, CUDA and install the requirements described in `requirements.txt` + run `pip install .` at the root of the repository. Everything is set up on Apollo server at FI, for more details see the last section of this document.


## Preparing dataset

You can receive access to the dataset after registration at ÚFAL.
Then, uncompress the data and run script for preparing the training data from raw dataset.

Czeng comes as a list of translated sentences. For training we would like to have training samples with multiple sentences. Ideally, we want to have full document, but split it into multiple training examples such that each fits into the model (transformer) input size.

This is what `translator generate-dataset` command does. It also filters out weird sentences (low score from ÚFAL language models or from ÚFAL translator model) or sentences which on its own cannot fit into model input size. It also filters out all rows from the source dataset which have ill formatting (typically when unescaped separator occurs in the text).

```
translator generate-dataset path/to/czeng20-train \
    --output-src=output-path-for-cs-sentences.txt \
    --output-tgt=output-path-for-en-sentences.txt \
    --on-bad-lines=skip
```

You can also customize the behaviour by changing some default parameters - run `translator generate-dataset --help` to see them.

It will utilize all cores for data processing and write parts of the training data separate text files. You will likely want to have one file
for czech sentences and one for english, so you need to concatanate the files:
```
cat output-path-for-cs-sentences.txt-* > output-path-for-cs-sentences.txt
cat output-path-for-en-sentences.txt-* > output-path-for-en-sentences.txt
```

Now, you have a clean dataset of paralel documents which all fit into model input size. Note that the default arguments dramatically reduces the size of the training dataset, so if you need more training data, you can change them (or use the synthetically translated monolingual data)

In my case, even the small subset is large enough - it is 4.5 million documents. The processed files are located at apollo in `nlp/projekty/pv061_xkadlci2/czeng/prepared/`.


## Training

I finetuned a pretrained neural translator **Helsinki-NLP/opus-mt-cs-en** released by University of Helsinki in Huggingface transformers library. I used an open source library [Adaptor](https://github.com/gaussalgo/adaptor), which is a lightweight wrapper around Huggingface transformes and provides a convenient way of loading the training data.

The training can be run using `translator train` command. I trained on an Nvidia A100 gpu so my training parameters were:  
```
translator train path/for/saving/model_checkopints \
    data/path-for-cs-sentences.txt \
    data/path-for-en-sentences.txt \
    data/path-for-validation-cs-sentences.txt \
    data/path-for-validation-en-sentences.txt \
    --epochs=10
    --batch-size=32
    --grad-acc-steps=4
    --save-steps=400
    --eval-steps=50000
    --lr=0.00002
    --gpu=0
```
Make sure to modify hyperparameters adequate for your setup (especially batch size). Parameter gpu specifies the id of gpu device used for training, if left unset, CPU is used.


## Inference

Inference can be done using `translator translate` command, which supports two modes. In either case, you must specify the path to the model checkpoint (but valid huggingface URL should work too). You should also specify a gpu id if you want to use one.

If you do not specify any other parameter, translator runs in interactive mode. It repeatedly asks for your input and outputs a translation. Example:
```
translator translate models/v1/Sequence2Sequence/ --gpu=0
>>> loading model
$$$ Type a czech sentence (leave empty to exit): Ahoj, co bude k obědu?
>>> Hey, what's for lunch?
```

The second mode is for translating text files. Keep in mind that it considers each line as a separate document and translates each line independently (and truncates a line if it is too long.) Example:

```
translator translate models/v1/Sequence2Sequence/
    --files data/some-file.txt
    --read-first-n=10
    --gpu=0
```
This writes a translation of first 10 lines of the file into `data/some-file.txt-translation`. You can check out parameters to change the default behaviour using `translator translate --help`.


## Evaluation

Use `translator translate` to generate translation of text file. Then you can run
```
translator eval \
    --actual=path/to/actual-translation.txt \
    --reference=path/to/reference-translation.txt
```
which computes the evaluation metrics BLEU, ROGUE-L and Meteor. Again, one line counts as one document. This can be used to compare various checkpoints of the model.

The filtered and processed test dataset contains around 30,000 documents. Translating this many documents would take over 5h on a fast gpu. To make evaluation faster, I further reduced the test set to 6,000 documents by taking the prepared test dataset and sampling it with pandas `df.sample(6000, random_state=42)` and saving it to `czeng/prepared/prepared-train-mini-src.txt` and `czeng/prepared/prepared-train-mini-tgt.txt`.


## Results

I compared multiple checkpoints on the mini test set. The checkpoint number indicates the number of gradient updates. In my setup, one epoch corresponds to approximately 35,000 gradient updates.

| checkpoint        | BLEU    | ROUGE-L | METEOR |
|-------------------|---------|---------|--------|
| pretrained model  | 32.1923 | 0.5765  | 0.5801 |
| checkpoint 15600  | 42.9991 | 0.6668  | 0.6638 |
| checkpoint 30000  | 43.2755 | 0.6701  | 0.6660 |
| checkpoint 45600  | 43.4970 | 0.6706  | 0.6644 |
| checkpoint 60000  | 43.8216 | 0.6739  | 0.6665 |
| checkpoint 75600  | 43.8190 | 0.6729  | 0.6661 |
| checkpoint 81600  | 44.2114 | 0.6744  | 0.6690 |

As we can see, most of the progress came from the first 15000 gradient updates (half of the first epoch). Additional training still increased the scores, but the improvements are negligible in comparison.


## Technical aspects on Apollo server

The project was realized on Apollo server at FI MUNI. The project root is at `/nlp/projekty/pv061_xkadlci2/`.
There are several folders:
```
- czeng
- models
- py38
- translator-cs-en
```
`czeng` contains the data. `models` contains the chechpoints of the trained model. `py38` is a conda environment which has everything set up. `translator-cs-en` is a clone of this repository.

You can activate the environment by running
```
conda activate /nlp/projekty/pv061_xkadlci2/py38
```
assuming you have miniconda installed.

The environment contains a CUDA installation compatible with Apollo gpu's and all the neccessary python packages. It was created by ```conda create --prefix <path> python=3.8```. If you want to recreate this environment, you can just create a conda environment and `pip install` all python packages listed in requirements.

However you might encounter problems with a pytorch+cuda installation so that it is compatible with Apollo and A100 gpu. In that case, you should first install
```
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
and then the remaining requirements from `requirements.txt`.

Note that Apollo has 17 GPUs which is more than default installation of pytorch+cuda can handle. You should set
```
export CUDA_VISIBLE_DEVICES=...
```
to select the GPU you want to use for training or inference. On Apollo, use `CUDA_VISIBLE_DEVICES=0` to select A100.

