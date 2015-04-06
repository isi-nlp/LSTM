#! /bin/bash
rm -rf *.o
/opt/local/bin/g++-mp-4.5 -I/opt/local/include -std=c++0x -c arpa.cpp
/opt/local/bin/g++-mp-4.5 -I/opt/local/include -std=c++0x -c test_arpa.cpp
/opt/local/bin/g++-mp-4.5 -L/opt/local/lib -lboost_program_options-mt -lboost_regex-mt -lboost_random-mt -o test_arpa test_arpa.o arpa.o 
