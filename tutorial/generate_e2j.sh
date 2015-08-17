#! /bin/bash

../src/generateFromNetwork --greedy 1 --decoder_model_file $1 --encoder_model_file $2 --testing_sent_file e2j.test.generate.carmel --predicted_sequence_file greedy.out
