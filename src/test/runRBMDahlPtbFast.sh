#! /bin/bash

OMP_NUM_THREADS=4 ./neuralLMFast --embedding_dimension 100 --n_vocab 10000 --train_file ptb.words.all.formatted.3grams.train.1004000 --ngram_size 3 --unigram_probs_file ptb.words.all.formatted.unigram.probs --minibatch_size 1000 --n_hidden 100  --learning_rate 0.01 --num_epochs 3 --words_file ptb.words.all.formatted.words.lst --embeddings_prefix embeddings.cpp.ptb-all.test --use_momentum 0 --validation_file ptb.words.all.formatted.validation --n_threads 4 --num_noise_samples 25 --L2_reg 0. --normalization_init 10

