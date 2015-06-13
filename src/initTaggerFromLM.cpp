#include <ctime>
#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdlib.h>


#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>
# include <boost/interprocess/managed_shared_memory.hpp>
# include <boost/interprocess/allocators/allocator.hpp>
# include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

//#include "fastonebigheader.h"
#include "model.h"
#include "propagator.h"
#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "multinomial.h"

//#include "gradientCheck.h"

//#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost;
using namespace boost::random;

using namespace nplm;

namespace ip = boost::interprocess;
typedef unordered_map<Matrix<int,Dynamic,1>, double> vector_map;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;


//typedef long long int data_size_t; // training data can easily exceed 2G instances

int main(int argc, char** argv)
{ 
	srand (time(NULL));
	setprecision(16);
    ios::sync_with_stdio(false);
    bool use_mmap_file, randomize;
    param myParam;
	
      // program options //
      CmdLine cmd("Tests a LSTM neural probabilistic language model.", ' ' , "0.3\n","");

      // The options are printed in reverse order


      ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 100. All gates, cells, hidden layers, \n \
		  							input and output embedding dimension are set to this value", false, 100, "int", cmd);
  	  //ValueArg<int> output_vocab_size("", "output_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<string> lm_model_file("", "lm_model_file", "Language Model file.", false, "", "string", cmd);
	  ValueArg<string> tagger_model_file("", "tagger_model_file", "Tagger Model file.", false, "", "string", cmd);

	  ValueArg<string> input_embeddings_file("","input_embeddings_file", "Read the input embeddings from the specified file. Default: none", false,"","string",cmd);
      ValueArg<double> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.01", false, 0.01, "double", cmd);
      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0.", false, 0, "bool", cmd);
      ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);	  
      cmd.parse(argc, argv);

    ///// Create and initialize the neural network and associated propagators.
    model nn;
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL
	int arg_num_hidden = num_hidden.getValue();
	string arg_lm_model_file = lm_model_file.getValue();
	string arg_tagger_model_file = tagger_model_file.getValue();
	string arg_input_embeddings_file = input_embeddings_file.getValue();
	
    if (arg_lm_model_file != ""){
      nn.read(myParam.model_file);
      cerr<<"reading the model"<<endl;
    } else {
		cerr<<"the model file has to be specified!"<<endl;
		exit(1);
    }
	
	if (input_embeddings_file.getValue() != ""){
		nn.W_x_to_i.read(myParam.input_embeddings_file);
		nn.W_x_to_f.read(myParam.input_embeddings_file);
		nn.W_x_to_c.read(myParam.input_embeddings_file);
		nn.W_x_to_o.read(myParam.input_embeddings_file);
	}
	int input_vocab_size =0;
	int output_vocab_size = 0;
	
    vector<string> input_words;
    if (input_words_file.getValue() != "")
    {
        readWordsFile(input_words_file.getValue(), input_words);	
		if (input_vocab_size == 0)
		    input_vocab_size = input_words.size();
    }

    vector<string> output_words;
    if (myParam.output_words_file != "")
    {
        readWordsFile(output_words_file.getValue(), output_words);
		if (output_vocab_size == 0)
		    output_vocab_size = output_words.size();
    }
	
	cerr<<"Input vocab size is "<<input_vocab_size<<endl;
	cerr<<"Output vocab size is "<<output_vocab_size<<endl;	
	
    //unsigned seed = std::time(0);
    unsigned seed = 1234; //for testing only
    mt19937 rng(seed);
	string parameter_update = "SGD";
	//Now initalizing the output layer of the NN
	nn.output_layer.resize(output_vocab_size,arg_num_hidden);
	nn.output_layer.initialize(rng,
	        init_normal.getValue(),
	        init_range.getValue(),
	        0.,
	        parameter_update,
	        0);	
	nn.write(arg_tagger_model_file,input_words,output_words);
    return 0;
}

