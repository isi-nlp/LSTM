#! /bin/bash

../src/generateFromNetwork --decoder_model_file $1 --encoder_model_file $2 --score 1 --run_lm 1 --testing_sent_file inferno.test.txt.carmel.lm --minibatch_size 20
