#! /bin/bash

/opt/local/bin/g++-mp-4.5 -O0 -fopenmp -m64 neuralLMHiddenBengioRectifiedLinear.cpp -I/Users/avaswani/boost/boost_1_52_0/boost -I. -I./tclap1.1.0/include/ -o neuralLMHiddenBengioRectifiedLinear


echo "compiled"

