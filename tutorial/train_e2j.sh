#! /bin/bash

../src/trainNeuralNetwork --training_sent_file e2j.train.carmel --validation_sent_file e2j.valid.carmel --learning_rate 1. --norm_threshold 0.75 --num_hidden 100 --model_prefix lstm.s2s.e2j --minibatch_size 30 --validation_minibatch_size 10 --init_range 0.1 --num_epochs 30


