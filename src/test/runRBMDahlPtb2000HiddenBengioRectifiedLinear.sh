#! /bin/bash

export OMP_NUM_THREADS=2 ;./neuralLMHiddenBengioRectifiedLinear --embedding_dimension 100 --n_vocab 5000 --train_file ptb.words.all.formatted.2000.3grams.train --ngram_size 3 --unigram_probs_file ptb.words.all.formatted.2000.unigram.probs --minibatch_size 1000 --n_hidden 50  --learning_rate 0.01 --num_epochs 10 --words_file ptb.words.all.formatted.2000.words.lst --embeddings_prefix embeddings.cpp.ptb-all.test --use_momentum 0 --validation_file ptb.words.all.formatted.2000.validation --n_threads 2  --num_noise_samples 5  --L2_reg 0.0000 --normalization_init 6 --validation_minibatch_size 100 

