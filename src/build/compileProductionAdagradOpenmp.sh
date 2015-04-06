#! /bin/bash

/opt/local/bin/g++-mp-4.5 -I/Users/avaswani/boost/boost_1_52_0/boost -fopenmp -DSERIAL1 -DEIGEN_NO_DEBUG -Wall -Wextra -O2 -m64 neuralLMAdagrad.cpp -I. -I./tclap1.1.0/include/ -o neuralLMAdagrad


echo "compiled"
