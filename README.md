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

Although they look panicking, the code provided by Stanford already completes most of the above files except `utils.py`, `model_embeddings.py`, and `nmt_model.py`, which only leaves a little space for you to fill in.

## Todos
 - Learn NLP from Stanford's [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) and [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) materials.
 - Build a question answering system for competitions.

## License
[PyTorch Attentional LSTM Machine Translation](https://github.com/yungshun317/pytorch-attentional-lstm-machine-translation) is released under the [MIT License](https://opensource.org/licenses/MIT) by [yungshun317](https://github.com/yungshun317).