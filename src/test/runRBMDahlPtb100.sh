#! /bin/bash

export OMP_NUM_THREADS=1 ;./neuralLM --embedding_dimension 100 --n_vocab 1039  --train_file ptb.words.all.formatted.100.3grams.train --ngram_size 3 --unigram_probs_file ptb.words.all.formatted.100.unigram.probs --minibatch_size 100 --n_hidden 50  --learning_rate 0.01 --num_epochs 200 --words_file ptb.words.all.formatted.100.words.lst --embeddings_prefix embeddings.cpp.ptb-all.test --use_momentum 0 --validation_file ptb.words.all.formatted.100.validation --n_threads 1  --num_noise_samples 100  --L2_reg 0.0000 --normalization_init 5 --validation_minibatch_size 100

