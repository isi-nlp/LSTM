#! /bin/bash

../src/trainNeuralNetwork --training_sent_file inferno.train.txt.carmel.lm --validation_sent_file inferno.valid.txt.carmel.lm --learning_rate 1. --norm_threshold 5 --model_prefix lstm.lm.inferno --minibatch_size 20 --validation_minibatch_size 20 --init_range 0.5 --num_hidden 10 --run_lm 1


