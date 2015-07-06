!# /bin/bash

./trainNeuralNetwork  --output_sent_file ../example/$1 --input_sent_file ../example/$2 --output_validation_sent_file ../example/$1 --input_validation_sent_file ../example/$2 --input_words_file ../example/input.words --output_words_file ../example/output.words --num_hidden 5 --num_epochs $3 --minibatch_size 1 --init_range 1. --learning_rate 1.5 --init_forget 0 --L2_reg $4 --gradient_check $5 --training_sequence_cont_file ../example/$2.cont --validation_sequence_cont_file ../example/$2.cont --model_prefix lstm --validation_minibatch_size 1
