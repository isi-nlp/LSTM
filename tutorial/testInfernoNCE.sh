#! /bin/bash

../src/generateFromNetwork --score 1 --run_lm 1 --decoder_model_file training.inferno.nce.decoder.best --testing_sent_file inferno.txt.test.lm
