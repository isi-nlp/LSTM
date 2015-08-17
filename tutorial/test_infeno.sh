#! /bin/bash

../src/generateFromNetwork --decoder_model_file lstm.lm.inferno.decoder.10 --encoder_model_file lstm.lm.inferno.encoder.10 --score 1 --run_lm 1 --testing_sent_file inferno.test.txt.carmel.lm --minibatch_size 20
