#! /bin/bash

../src/trainNeuralNetwork --training_sent_file inferno.txt.train.lm --model_prefix training.inferno.log --num_hidden 50 --init_range 0.1 --norm_threshold 0.5 --learning_rate 1 --validation_sent_file inferno.txt.valid.lm --num_epochs 10 --minibatch_size 16 --loss_function log --run_lm 1
