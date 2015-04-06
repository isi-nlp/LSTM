#include <stdio.h>
#include <iomanip>

#include <iostream>
#include <list>
#include <ctime>
#include <cstdio>

#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>
#include "param.h"

#define EIGEN_DONT_PARALLELIZE
using namespace std;
using namespace TCLAP;
using namespace boost::random;




void readParams(param &myParam,int argc, char** argv)
{
    try{
      // program options //
      CmdLine cmd("Command description message ", ' ' , "1.0");

      ValueArg<string> train_file("", "train_file", "training file" , true, "string", "string", cmd);

      ValueArg<int> num_tag_types("", "num_tag_types", "The number of tags. Default: 45", false, 45, "int", cmd);
      ValueArg<int> num_words("", "num_words", "The number of words. Default: 1000", false, 1000, "int", cmd);
      ValueArg<float> smoothing("", "smoothing", "Smoothing : 5e-05", false, 5e-05, "int", cmd);

      ValueArg<string> transition_words_file("", "transition_words_file", "transition words file", true,"string", "string", cmd);

      ValueArg<char> activation("", "activation", "tanh (t) or sigmoid (s) activation. Default: t", falst,'t', "char", cmd);

      ValueArg<int> n_hidden("", "n_hidden", "The number of nodes in the hidden layer. Default: 1000", false, 1000, "int", cmd);

      ValueArg<int> n_input("", "n_input", "The number of nodes in the input layer. Default: 50(for embedding) + 45(for tags)", false, 95, "int", cmd);

      ValueArg<int> minibatch_size("", "minibatch_size", "The minibatch size. Default: 64", false, 64, "int", cmd);


      cmd.parse(argc, argv);

      // define program parameters //
      myParam.train_file = train_file.getValue();
      myParam.num_tag_types= num_tag_types.getValue();
      myParam.num_words= num_words.getValue();
      myParam.smoothing = smoothing.getValue();
      myParam.transition_words_file = transition_words_file.getValue();
      myParam.n_hidden = n_hidden.getValue();
      myParam.n_input = n_input.getValue();
      myParam.minibatch_size = minibatch_size.getValue();
      myParam.activation = activation.getValue();
      // print program command to stdout//

      cout << "Command line: " << endl;

      for (int i = 0; i < argc; i++)
      {
        cout << argv[i] << endl;
      }
      cout << endl;

      cout << train_file.getDescription() << " : " << train_file.getValue() << endl;
      cout << num_tag_types.getDescription() << " : " << num_tag_types.getValue() << endl;
      cout << num_words.getDescription() << " : " << num_words.getValue() << endl;

    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

}

