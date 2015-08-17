#! /bin/bash

../src/generateFromNetwork --decoder_model_file $1 --encoder_model_file $2 --testing_sent_file e2j.test.score.carmel --score 1
