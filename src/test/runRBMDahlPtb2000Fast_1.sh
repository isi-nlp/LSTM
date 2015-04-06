#! /bin/bash

export OMP_NUM_THREADS=1 ;./neuralLMFast --embedding_dimension 50 --n_vocab 5000 --train_file ptb.words.all.formatted.2000.3grams.train --ngram_size 3 --unigram_probs_file ptb.words.all.formatted.2000.unigram.probs --minibatch_size 1000 --n_hidden 100 --learning_rate 0.05 --num_epochs 1 --words_file ptb.words.all.formatted.2000.words.lst --embeddings_prefix embeddings.cpp.ptb-all.test --use_momentum 0 --validation_file ptb.words.all.formatted.2000.validation --n_threads 1 --num_noise_samples 25  --L2_reg 0. --normalization_init 5 --validation_minibatch_size 5000

