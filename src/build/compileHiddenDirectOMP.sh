#! /bin/bash

/opt/local/bin/g++-mp-4.5 -O0 -fopenmp -m64 neuralLMHiddenDirect.cpp -I/Users/avaswani/boost/boost_1_52_0/boost -I. -I./tclap1.1.0/include/ -o neuralLMHiddenDirect


echo "compiled"

