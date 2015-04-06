#! /bin/bash

export OMP_NUM_THREADS=4 ; ./neuralLM --embedding_dimension 50 --n_vocab 1000 --train_file test.words.new-formatted.24k.3grams.train.23000 --ngram_size 3 --unigram_probs_file test.words.new-formatted.24k.unigram.probs --minibatch_size 100 --n_hidden 100 --learning_rate 0.1 --num_epochs 2 --words_file test.words.new-formatted.24k.word_list --embeddings_prefix embeddings.cpp.test --use_momentum 1 --validation_file test.words.new-formatted.24k.3grams.validation --n_threads 4 --num_noise_samples 25 --initial_momentum 0.5 --final_momentum 0.9 --L2_reg 0.000 --normalization_init 5 --validation_minibatch_size 100


