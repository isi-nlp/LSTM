#! /bin/bash

/opt/local/bin/g++-mp-4.5 -std=c++0x -O0 -fopenmp -DEIGEN_DONT_PARALLELIZE -m64 eigenTest.cpp -I/Users/avaswani/boost/boost_1_52_0/boost -I. -I./tclap1.1.0/include/ -o eigenTest


echo "compiled"

