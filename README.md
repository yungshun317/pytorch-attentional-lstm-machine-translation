# PyTorch Attentional LSTM Machine Translation

[![Made with Python](https://img.shields.io/badge/Made_with-Python-blue.svg)](https://img.shields.io/badge/Made_with-Python-blue.svg) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a sequence-to-sequence (Seq2Seq) neural network with attention to build a Neural Machine Translation (NMT) system. It is mainly based on the assignment of the [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) course from Stanford.

## Overview

Here is the project structure:
* `__init__.py`.
* `run.py`: runs `train`, which trains the NMT model, or `decode`, which performs decoding on a test set and save the best-scoring decoding results. If the target gold-standard sentences are given, the `decode` function also computes corpus-level BLEU score.
* `sanity_check.py`: a basic test to quickly evaluate whether a claim or the result of a calculation can possibly be true.
* `vocab.py`: generates vocabulary with the `Vocab` and `VocabEntry` classes.
* `utils.py`: the utility methods, including `pad_sents`, `read_corpus`, and `batch_iter`.
* `model_embeddings.py`: a class that converts input words to their embeddings.
* `nmt_model.py`: implements a simple neural machine translation model.
    * `__init__`: initializes the necessary model embeddings (the `ModelEmbeddings` class from `model_embeddings.py`) and layers (LSTM, projection, and dropout) for the NMT system.
    * `encode`: converts the padded sentences into the tensor `X` generating `enc_hiddens` and `dec_init_state` for the decoder.
    * `decode`: constructs `Y` and runs the `step` function over every timestep for the input.
    * `step`: applies the decoder's LSTM cell for a single timestep, computing the encoding of the target word `dec_state`, the attention scores `e_t`, the attention distribution `alpha_t`, the attentional output `a_t`, and finally the combined output `O_t`.

Although they look panicking, the code provided by Stanford already completes most of the above files except `utils.py` (`pad_sents`), `model_embeddings.py` (`__init__`), and `nmt_model.py` (`__init__`, `encode`, `decode`, `step`), which only leaves a little space for us to fill in. Also you had better take reference of [the handout](http://web.stanford.edu/class/cs224n/assignments/a4.pdf) for clarification.

## Up & Running

Test the `encode` implementation.
```sh
$ python sanity_check.py 1d
Running Sanity Check for Question 1d: Encode
--------------------------------------------------------------------------------
enc_hiddens Sanity Checks Passed!
dec_init_state[0] Sanity Checks Passed!
dec_init_state[1] Sanity Checks Passed!
--------------------------------------------------------------------------------
All Sanity Checks Passed for Question 1d: Encode!
--------------------------------------------------------------------------------
```

Test the `decode` implementation.
```sh
$ python sanity_check.py 1e
--------------------------------------------------------------------------------
Running Sanity Check for Question 1e: Decode
--------------------------------------------------------------------------------
combined_outputs Sanity Checks Passed!
--------------------------------------------------------------------------------
All Sanity Checks Passed for Question 1e: Decode!
--------------------------------------------------------------------------------
```

Test the `step` implementation.
```sh
$ python sanity_check.py 1f
--------------------------------------------------------------------------------
Running Sanity Check for Question 1f: Step
--------------------------------------------------------------------------------
..\aten\src\ATen\native\LegacyDefinitions.cpp:14: UserWarning: masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated,please use a mask with dtype torch.bool instead.
dec_state[0] Sanity Checks Passed!
dec_state[1] Sanity Checks Passed!
combined_output  Sanity Checks Passed!
e_t Sanity Checks Passed!
--------------------------------------------------------------------------------
All Sanity Checks Passed for Question 1f: Step!
--------------------------------------------------------------------------------
```

Generate the necessary vocab file.
```sh
$ python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
read in source sentences: ./en_es_data/train.es
read in target sentences: ./en_es_data/train.en
initialize source vocabulary ..
number of word types: 172418, number of word types w/ frequency >= 2: 80623
initialize target vocabulary ..
number of word types: 128873, number of word types w/ frequency >= 2: 64215
generated vocabulary, source 50004 words, target 50002 words
vocabulary saved to vocab.json
```

Train the model with GPU support.
```sh
$ python -W ignore run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
...
epoch 13, iter 88000, cum. loss 28.77, cum. ppl 5.11 cum. examples 64000
begin validation ...
validation: iter 88000, dev. ppl 17.137217
hit patience 5
hit #5 trial
early stop!
```
Because masking via `torch.uint8` Tensors is now deprecated in favor of masking via `torch.bool` Tensors in PyTorch 1.2.0, the program will generate tons of warnings during training. I add `-W ignore` to hide these warnings.

Test the model.
```sh
$ python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
...
Corpus BLEU: 22.65632218633906
```
We have model's corpus BLEU Score larger than 21 which meets the requirement.  

## Todos
 - Learn NLP from Stanford's [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) and [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) materials.
 - Build a question answering system for competitions.

## License
[PyTorch Attentional LSTM Machine Translation](https://github.com/yungshun317/pytorch-attentional-lstm-machine-translation) is released under the [MIT License](https://opensource.org/licenses/MIT) by [yungshun317](https://github.com/yungshun317).