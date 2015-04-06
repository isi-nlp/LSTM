#! /bin/bash

export OMP_NUM_THREADS=2 ; ./neuralLM --embedding_dimension 2 --num_noise_samples 2 --n_vocab 4 --train_file train.test --ngram_size 3 --unigram_probs_file unigram.probs.test --minibatch_size 2 --n_hidden 100 --learning_rate 1 --num_epochs 20 --words_file words.lst.test --embeddings_prefix embeddings.cpp.test --use_momentum 0 --validation_file train.test --n_threads 1 --L2_reg 0 --normalization_init 1


