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

#include "fastonebigheader.h"
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
    try {
      // program options //
      CmdLine cmd("Tests a LSTM neural probabilistic language model.", ' ' , "0.3\n","");

      // The options are printed in reverse order

      ValueArg<string> unigram_probs_file("", "unigram_probs_file", "Unigram model (deprecated and ignored)." , false, "", "string", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);

      ValueArg<double> final_momentum("", "final_momentum", "Final value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
      ValueArg<double> initial_momentum("", "initial_momentum", "Initial value of momentum. Default: 0.9.", false, 0.9, "double", cmd);
      ValueArg<bool> use_momentum("", "use_momentum", "Use momentum (hidden layer weights only). 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

      ValueArg<double> normalization_init("", "normalization_init", "Initial normalization parameter. Default: 0.", false, 0.0, "double", cmd);
      ValueArg<bool> normalization("", "normalization", "Learn individual normalization factors during training. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);

      ValueArg<bool> mmap_file("", "mmap_file", "Use memory mapped files. This is useful if the entire data cannot fit in memory. prepareNeuralLM can generate memory mapped files", false, 0, "bool", cmd);

      ValueArg<bool> arg_randomize("", "randomize", "Randomize training instances for better training. 1 = yes, 0 = no. Default: 1.", false, true, "bool", cmd);

      ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 100.", false, 100, "int", cmd);

      ValueArg<double> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "double", cmd);

      ValueArg<double> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 1.", false, 1., "double", cmd);
	  ValueArg<double> fixed_partition_function("", "fixed_partition_function", "Fixed log normalization constant value. Default: 0.", false, 0., "double", cmd);

      ValueArg<double> conditioning_constant("", "conditioning_constant", "Constant to condition the RMS of the expected square of the gradient in ADADELTA. Default: 10E-3.", false, 10E-3, "double", cmd);

      ValueArg<double> decay("", "decay", "Decay for ADADELTA. Default: 0.95", false, 0.95, "double", cmd);
      ValueArg<double> adagrad_epsilon("", "adagrad_epsilon", "Constant to initialize the L2 squared norm of the gradients with.\
          Default: 10E-3", false, 10E-3, "double", cmd);
      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 64.", false, 64, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 1000.", false, 1000, "int", cmd);

      ValueArg<int> num_epochs("", "num_epochs", "Number of epochs. Default: 10.", false, 10, "int", cmd);

      ValueArg<double> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.01", false, 0.01, "double", cmd);
	  ValueArg<double> init_forget("", "init_forget", "value to initialize the bias of the forget gate. Default: 20", false, 20, "double", cmd);
      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0.", false, 0, "bool", cmd);

      ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: log.", false, "log", "string", cmd);
      ValueArg<string> activation_function("", "activation_function", "Activation function (identity, rectifier, tanh, hardtanh). Default: rectifier.", false, "rectifier", "string", cmd);
      ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 100. All gates, cells, hidden layers, \n \
		  							input and output embedding dimension are set to this value", false, 100, "int", cmd);
      ValueArg<bool> share_embeddings("", "share_embeddings", "Share input and output embeddings. 1 = yes, 0 = no. Default: 0.", false, 0, "bool", cmd);
      ValueArg<int> output_embedding_dimension("", "output_embedding_dimension", "Number of output embedding dimensions. Default: 50.", false, 50, "int", cmd);
      ValueArg<int> input_embedding_dimension("", "input_embedding_dimension", "Number of input embedding dimensions. Default: 50.", false, 50, "int", cmd);
      ValueArg<int> embedding_dimension("", "embedding_dimension", "Number of input and output embedding dimensions. Default: none.", false, -1, "int", cmd);

      ValueArg<int> vocab_size("", "vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> input_vocab_size("", "input_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> output_vocab_size("", "output_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      ValueArg<int> ngram_size("", "ngram_size", "Size of n-grams. Default: auto.", false, 0, "int", cmd);

      ValueArg<string> model_prefix("", "model_prefix", "Prefix for output model files." , false, "", "string", cmd);
      ValueArg<string> words_file("", "words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> parameter_update("", "parameter_update", "parameter update type.\n Stochastic Gradient Descent(SGD)\n \
          ADAGRAD(ADA)\n \
          ADADELTA(ADAD)" , false, "SGD", "string", cmd);
      ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);
	  ValueArg<string> input_sent_file("", "input_sent_file", "Input sentences file." , false, "", "string", cmd);
	  ValueArg<string> output_sent_file("", "output_sent_file", "Input sentences file." , false, "", "string", cmd);
	  ValueArg<string> training_sequence_cont_file("", "training_sequence_cont_file", "Training sequence continuation file" , false, "", "string", cmd);
	  ValueArg<string> validation_sequence_cont_file("", "validation_sequence_cont_file", "Validation sequence continuation file" , false, "", "string", cmd);
	  ValueArg<string> input_validation_sent_file("", "input_validation_sent_file", "Input sentences file." , false, "", "string", cmd);
	  ValueArg<string> output_validation_sent_file("", "output_validation_sent_file", "Input sentences file." , false, "", "string", cmd);	  
      ValueArg<string> validation_file("", "validation_file", "Validation data (one numberized example per line)." , false, "", "string", cmd);
	  ValueArg<bool> gradient_check("", "gradient_check", "Do you want to do a gradient check or not. 1 = Yes, 0 = No. Default: 0.", false, 0, "bool", cmd);
      //ValueArg<string> train_file("", "train_file", "Training data (one numberized example per line)." , true, "", "string", cmd);
	  ValueArg<bool> norm_clipping("", "norm_clipping", "Do you want to do norm clipping or gradient clipping. 1 = norm cilpping, \n \
		  			0 = gradient clipping. Default: 0.", false, 1, "bool", cmd);
	  ValueArg<bool> restart_states("", "restart_states", "If yes, then the hidden and cell values will be restarted after every minibatch \n \
		  Default: 1 = yes, \n \
		  			0 = gradient clipping. Default: 0.", false, 1, "bool", cmd);	  
      ValueArg<string> model_file("", "model_file", "Model file.", false, "", "string", cmd);
	  ValueArg<double> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "double", cmd);

      cmd.parse(argc, argv);

      // define program parameters //
      use_mmap_file = mmap_file.getValue();
      randomize = arg_randomize.getValue();
      myParam.model_file = model_file.getValue();
      //myParam.train_file = train_file.getValue();
      myParam.validation_file = validation_file.getValue();
      myParam.input_words_file = input_words_file.getValue();
      myParam.output_words_file = output_words_file.getValue();
	  myParam.input_sent_file = input_sent_file.getValue();
	  myParam.output_sent_file = output_sent_file.getValue();
	  myParam.input_validation_sent_file = input_validation_sent_file.getValue();
	  myParam.output_validation_sent_file = output_validation_sent_file.getValue();	  
	  myParam.training_sequence_cont_file = training_sequence_cont_file.getValue();
	  myParam.validation_sequence_cont_file = validation_sequence_cont_file.getValue();
	  
      if (words_file.getValue() != "")
	      myParam.input_words_file = myParam.output_words_file = words_file.getValue();

      myParam.model_prefix = model_prefix.getValue();

      myParam.ngram_size = ngram_size.getValue();
      myParam.vocab_size = vocab_size.getValue();
      myParam.input_vocab_size = input_vocab_size.getValue();
      myParam.output_vocab_size = output_vocab_size.getValue();
      if (vocab_size.getValue() >= 0) {
	      myParam.input_vocab_size = myParam.output_vocab_size = vocab_size.getValue();
      }
      myParam.num_hidden = num_hidden.getValue();
      myParam.activation_function = activation_function.getValue();
      myParam.loss_function = loss_function.getValue();

      myParam.num_threads = num_threads.getValue();

      myParam.num_noise_samples = num_noise_samples.getValue();

      myParam.input_embedding_dimension = input_embedding_dimension.getValue();
      myParam.output_embedding_dimension = output_embedding_dimension.getValue();
      if (embedding_dimension.getValue() >= 0) {
	      myParam.input_embedding_dimension = myParam.output_embedding_dimension = embedding_dimension.getValue();
      }

      myParam.minibatch_size = minibatch_size.getValue();
	  myParam.minibatch_size = 1;
      myParam.validation_minibatch_size = validation_minibatch_size.getValue();
      myParam.num_epochs= num_epochs.getValue();
      myParam.learning_rate = learning_rate.getValue();
      myParam.conditioning_constant = conditioning_constant.getValue();
      myParam.decay = decay.getValue();
      myParam.adagrad_epsilon = adagrad_epsilon.getValue();
	  myParam.fixed_partition_function = fixed_partition_function.getValue();
      myParam.use_momentum = use_momentum.getValue();
      myParam.share_embeddings = share_embeddings.getValue();
      myParam.normalization = normalization.getValue();
      myParam.initial_momentum = initial_momentum.getValue();
      myParam.final_momentum = final_momentum.getValue();
      myParam.L2_reg = L2_reg.getValue();
      myParam.init_normal= init_normal.getValue();
      myParam.init_range = init_range.getValue();
	  myParam.init_forget = init_forget.getValue();
      myParam.normalization_init = normalization_init.getValue();
      myParam.parameter_update = parameter_update.getValue();
	  myParam.gradient_check = gradient_check.getValue();
	  myParam.norm_clipping = norm_clipping.getValue();
	  myParam.norm_threshold = norm_threshold.getValue();
	  myParam.restart_states = norm_threshold.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
      //cerr << train_file.getDescription() << sep << train_file.getValue() << endl;
      cerr << validation_file.getDescription() << sep << validation_file.getValue() << endl;
      cerr << input_words_file.getDescription() << sep << input_words_file.getValue() << endl;
      cerr << output_words_file.getDescription() << sep << output_words_file.getValue() << endl;
      cerr << model_prefix.getDescription() << sep << model_prefix.getValue() << endl;

      cerr << ngram_size.getDescription() << sep << ngram_size.getValue() << endl;
      cerr << input_vocab_size.getDescription() << sep << input_vocab_size.getValue() << endl;
      cerr << output_vocab_size.getDescription() << sep << output_vocab_size.getValue() << endl;
      cerr << mmap_file.getDescription() << sep << mmap_file.getValue() << endl;
	  cerr << norm_clipping.getDescription() << sep << norm_clipping.getValue() <<endl;
	  cerr << norm_threshold.getDescription() << sep << norm_threshold.getValue() <<endl;
	  cerr << gradient_check.getDescription() <<sep <<gradient_check.getValue() <<endl;
	  cerr << restart_states.getDescription() <<sep <<restart_states.getValue() <<endl;
	  cerr << fixed_partition_function.getDescription() <<sep <<fixed_partition_function.getValue() <<endl;
	  

      if (embedding_dimension.getValue() >= 0)
      {
	      cerr << embedding_dimension.getDescription() << sep << embedding_dimension.getValue() << endl;
      }
      else
      {
	      cerr << input_embedding_dimension.getDescription() << sep << input_embedding_dimension.getValue() << endl;
	      cerr << output_embedding_dimension.getDescription() << sep << output_embedding_dimension.getValue() << endl;
      }
      cerr << share_embeddings.getDescription() << sep << share_embeddings.getValue() << endl;
      if (share_embeddings.getValue() && input_embedding_dimension.getValue() != output_embedding_dimension.getValue())
      {
	      cerr << "error: sharing input and output embeddings requires that input and output embeddings have same dimension" << endl;
	      exit(1);
      }

      cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;

      if (string_to_activation_function(activation_function.getValue()) == InvalidFunction)
      {
	      cerr << "error: invalid activation function: " << activation_function.getValue() << endl;
	      exit(1);
      }
      cerr << activation_function.getDescription() << sep << activation_function.getValue() << endl;

      if (string_to_loss_function(loss_function.getValue()) == InvalidLoss)
      {
	      cerr << "error: invalid loss function: " << loss_function.getValue() << endl;
	      exit(1);
      }
      cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;

      cerr << init_normal.getDescription() << sep << init_normal.getValue() << endl;
      cerr << init_range.getDescription() << sep << init_range.getValue() << endl;

      cerr << num_epochs.getDescription() << sep << num_epochs.getValue() << endl;
      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
      if (myParam.validation_file != "") {
	     cerr << validation_minibatch_size.getDescription() << sep << validation_minibatch_size.getValue() << endl;
      }
      cerr << learning_rate.getDescription() << sep << learning_rate.getValue() << endl;
      cerr << L2_reg.getDescription() << sep << L2_reg.getValue() << endl;

      cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() << endl;

      cerr << normalization.getDescription() << sep << normalization.getValue() << endl;
      if (myParam.normalization){
	      cerr << normalization_init.getDescription() << sep << normalization_init.getValue() << endl;
      }

      cerr << use_momentum.getDescription() << sep << use_momentum.getValue() << endl;
      if (myParam.use_momentum)
      {
        cerr << initial_momentum.getDescription() << sep << initial_momentum.getValue() << endl;
        cerr << final_momentum.getDescription() << sep << final_momentum.getValue() << endl;
      }

      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;

      if (unigram_probs_file.getValue() != "")
      {
	      cerr << "Note: --unigram_probs_file is deprecated and ignored." << endl;
      }
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);
    int save_threads;

    //unsigned seed = std::time(0);
    unsigned seed = 1234; //for testing only
    mt19937 rng(seed), rng_grad_check(seed);

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read training data

    vector<int> training_data_flat;
	vector< vector<int> > training_input_sent, validation_input_sent, training_sequence_cont_sent;
	vector< vector<int> > training_output_sent, validation_output_sent, validation_sequence_cont_sent;
	
    vec * training_data_flat_mmap;
    data_size_t training_data_size, validation_data_size; //num_tokens;
    ip::managed_mapped_file mmap_file;
	/*
    if (use_mmap_file == false) {
      cerr<<"Reading data from regular text file "<<endl;
      readDataFile(myParam.train_file, myParam.ngram_size, training_data_flat, myParam.minibatch_size);
      training_data_size = training_data_flat.size()/myParam.ngram_size;
    } else {
 
    }
	
	*/
	
	//Reading the input and output sent files
	data_size_t total_output_tokens,
				total_input_tokens,
				total_validation_input_tokens, 
				total_validation_output_tokens, 
				total_training_sequence_tokens,
				total_validation_sequence_tokens;
				
	total_output_tokens = 
		total_input_tokens = 
			total_validation_input_tokens = 
				total_validation_output_tokens = 
					total_training_sequence_tokens = 
						total_validation_sequence_tokens = 0;
	//data_size_t total_input_tokens = 0;
	//data_size
	readSentFile(myParam.input_sent_file, training_input_sent,myParam.minibatch_size, total_input_tokens);
	readSentFile(myParam.output_sent_file, training_output_sent,myParam.minibatch_size, total_output_tokens);
    readSentFile(myParam.training_sequence_cont_file, training_sequence_cont_sent, myParam.minibatch_size, total_training_sequence_tokens);
	
	training_data_size = training_input_sent.size();

	data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;

    cerr<<"Number of input tokens "<<total_input_tokens<<endl;
	cerr<<"Number of output tokens "<<total_output_tokens<<endl;
	cerr<<"Number of minibatches "<<num_batches<<endl;
    //data_size_t training_data_size = num_tokens / myParam.ngram_size;
	
    cerr << "Number of training instances "<< training_input_sent.size() << endl;
    cerr << "Number of validation instances "<< validation_input_sent.size() << endl;
	
    Matrix<int,Dynamic,Dynamic> training_data;
	Matrix<int,Dynamic,Dynamic> training_input_sent_data, training_output_sent_data;
	Matrix<int,Dynamic,Dynamic> validation_input_sent_data, validation_output_sent_data;
	Array<int,Dynamic,Dynamic> validation_sequence_cont_sent_data, training_sequence_cont_sent_data;
    //(training_data_flat.data(), myParam.ngram_size, training_data_size);
    
	
    if (use_mmap_file == false) {
      training_data = Map< Matrix<int,Dynamic,Dynamic> >(training_data_flat.data(), myParam.ngram_size, training_data_size);
    }

    // If neither --input_vocab_size nor --input_words_file is given, set input_vocab_size to the maximum word index
    if (myParam.input_vocab_size == 0 and myParam.input_words_file == "")
    {
        myParam.input_vocab_size = training_data.topRows(myParam.ngram_size-1).maxCoeff()+1;
    }

    // If neither --output_vocab_size nor --output_words_file is given, set output_vocab_size to the maximum word index
    if (myParam.output_vocab_size == 0 and myParam.output_words_file == "")
    {
        myParam.output_vocab_size = training_data.row(myParam.ngram_size-1).maxCoeff()+1;
    }
	/*
    if (use_mmap_file == false && randomize == true) {
      cerr<<"Randomly shuffling data..."<<endl;
      // Randomly shuffle training data to improve learning
      for (data_size_t i=training_data_size-1; i>0; i--)
      {
        data_size_t j = uniform_int_distribution<data_size_t>(0, i-1)(rng);
        training_data.col(i).swap(training_data.col(j));
      }
    }
	*/
	
	
    // Read validation data
    //vector<int> validation_data_flat;
    validation_data_size = 0;
    
	/*
    if (myParam.validation_file != "")
    {
      readDataFile(myParam.validation_file, myParam.ngram_size, validation_data_flat);
      validation_data_size = validation_data_flat.size() / myParam.ngram_size;
      cerr << "Number of validation instances: " << validation_data_size << endl;
    }
	*/
	
	if (myParam.input_validation_sent_file != ""){
		readSentFile(myParam.input_validation_sent_file, 
		validation_input_sent,
		myParam.validation_minibatch_size, 
		total_validation_input_tokens);
		
		readSentFile(myParam.output_validation_sent_file, 
		validation_output_sent,
		myParam.validation_minibatch_size, 
		total_validation_output_tokens);
		
		readSentFile(myParam.validation_sequence_cont_file, 
			validation_sequence_cont_sent,
			myParam.validation_minibatch_size, 
			total_validation_sequence_tokens);			
	}
	
	cerr<<"Validation input tokens "<<total_validation_input_tokens<<endl;
	cerr<<"Validation output tokens "<<total_validation_output_tokens<<endl;
	validation_data_size = validation_input_sent.size();
	//data_size_t num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size + 1;
    //Map< Matrix<int,Dynamic,Dynamic> > validation_data(validation_data_flat.data(), myParam.ngram_size, validation_data_size);

    ///// Read in vocabulary file. We don't actually use it; it just gets reproduced in the output file

    vector<string> input_words;
    if (myParam.input_words_file != "")
    {
        readWordsFile(myParam.input_words_file, input_words);
	if (myParam.input_vocab_size == 0)
	    myParam.input_vocab_size = input_words.size();
    }

    vector<string> output_words;
    if (myParam.output_words_file != "")
    {
        readWordsFile(myParam.output_words_file, output_words);
	if (myParam.output_vocab_size == 0)
	    myParam.output_vocab_size = output_words.size();
    }
	cerr<<"Input vocab size is "<<myParam.input_vocab_size<<endl;
	cerr<<"Output vocab size is "<<myParam.output_vocab_size<<endl;
	
    ///// Construct unigram model and sampler that will be used for NCE

    vector<data_size_t> unigram_counts(myParam.output_vocab_size);
    for (data_size_t train_id=0; train_id < training_output_sent.size(); train_id++)
    {
		for (int j=0; j<training_output_sent[train_id].size(); j++) {
			int output_word = training_output_sent[train_id][j];
			unigram_counts[output_word] += 1;
		}
    }
	/*
	for (int i=0; i<unigram_counts.size(); i++)
		cerr<<"The count of word "<<i<<" is "<<unigram_counts[i]<<endl;
	*/
    multinomial<data_size_t> unigram (unigram_counts);
	/*
	//generating 10 noise samples for testing
	for (int i=0; i<20; i++){
		cerr<<"A sample is "<<unigram.sample(rng)<<endl;
	}
	*/
    ///// Create and initialize the neural network and associated propagators.
	myParam.input_embedding_dimension = myParam.num_hidden;
	myParam.output_embedding_dimension = myParam.num_hidden;
	
    model nn;
    nn.resize(myParam.ngram_size,
        myParam.input_vocab_size,
        myParam.output_vocab_size,
        myParam.input_embedding_dimension,
        myParam.num_hidden,
        myParam.output_embedding_dimension);

    nn.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
		-log(myParam.output_vocab_size),
        myParam.init_forget,
        myParam.parameter_update,
        myParam.adagrad_epsilon);	
	//Creating the input node
	google_input_model input(myParam.num_hidden, 
						myParam.input_vocab_size,
						myParam.input_embedding_dimension);
	input.resize(myParam.input_vocab_size,
	    myParam.input_embedding_dimension,
	    myParam.num_hidden);

	input.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
        myParam.parameter_update,
        myParam.adagrad_epsilon);
	
	nn.set_input(input);
				
	model nn_decoder;
	    nn_decoder.resize(myParam.ngram_size,
	        myParam.output_vocab_size,
	        myParam.output_vocab_size,
	        myParam.input_embedding_dimension,
	        myParam.num_hidden,
	        myParam.output_embedding_dimension);

	    nn_decoder.initialize(rng,
	        myParam.init_normal,
	        myParam.init_range,
			-log(myParam.output_vocab_size),
	        myParam.init_forget,
	        myParam.parameter_update,
	        myParam.adagrad_epsilon);		
		//Creating the input node
		google_input_model decoder_input(myParam.num_hidden, 
							myParam.input_vocab_size,
							myParam.input_embedding_dimension);
		decoder_input.resize(myParam.input_vocab_size,
		    myParam.input_embedding_dimension,
		    myParam.num_hidden);

		decoder_input.initialize(rng,
	        myParam.init_normal,
	        myParam.init_range,
	        myParam.parameter_update,
	        myParam.adagrad_epsilon);

		nn_decoder.set_input(decoder_input);	
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL


	
    if (myParam.model_file != ""){
      nn.read(myParam.model_file);
      cerr<<"reading the model"<<endl;
    } else {
      nn.set_activation_function(string_to_activation_function(myParam.activation_function));
	  rng_grad_check = rng; //The range for gradient check should have exactly the same state as rng for the NCE gradient checking to work
	  
    }
    loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

    propagator<Google_input_node, google_input_model> prop(nn, nn_decoder, myParam.minibatch_size);
	//IF we're using NCE, then the minibatches have different sizes
	if (loss_function == NCELoss)
		prop.resizeNCE(myParam.num_noise_samples, myParam.fixed_partition_function);
    propagator<Google_input_node, google_input_model> prop_validation(nn, nn_decoder, myParam.validation_minibatch_size);
	//if (loss_function == NCELoss){
	//	propagator.
	//}
    SoftmaxNCELoss<multinomial<data_size_t> > softmax_nce_loss(unigram);
    // normalization parameters
    //vector_map c_h, c_h_running_gradient;
    
    ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////


    cerr<<"Number of training minibatches: "<<num_batches<<endl;
	
	
    int num_validation_batches = 0;
    if (validation_data_size > 0)
    {
        num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
		cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
    } 
	
    double current_momentum = myParam.initial_momentum;
    double momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
    double current_learning_rate = myParam.learning_rate;
    double current_validation_ll = 0.0;

    int ngram_size = myParam.ngram_size;
    int input_vocab_size = myParam.input_vocab_size;
    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
    int validation_minibatch_size = myParam.validation_minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;

	/*
    if (myParam.normalization)
    {
      for (data_size_t i=0;i<training_data_size;i++)
      {
          Matrix<int,Dynamic,1> context = training_data.block(0,i,ngram_size-1,1);
          if (c_h.find(context) == c_h.end())
          {
              c_h[context] = -myParam.normalization_init;
          }
      }
    }
	*/
	
	//Resetting the gradient in the beginning
	prop.resetGradient();
    for (int epoch=0; epoch<myParam.num_epochs; epoch++)
    { 
        cerr << "Epoch " << epoch+1 << endl;
        cerr << "Current learning rate: " << current_learning_rate << endl;

        if (myParam.use_momentum) 
	    cerr << "Current momentum: " << current_momentum << endl;
	else
            current_momentum = -1;

	cerr << "Training minibatches: ";

	double log_likelihood = 0.0;

	int num_samples = 0;
	if (loss_function == LogLoss)
	    num_samples = output_vocab_size;
	else if (loss_function == NCELoss)
	    num_samples = 1+num_noise_samples;
	//Generating 10 samples
	
	/*
	Matrix<double,Dynamic,Dynamic> minibatch_weights(num_samples, minibatch_size);
	Matrix<int,Dynamic,Dynamic> minibatch_samples(num_samples, minibatch_size);
	Matrix<double,Dynamic,Dynamic> scores(num_samples, minibatch_size);
	Matrix<double,Dynamic,Dynamic> probs(num_samples, minibatch_size);
	*/
	
	//cerr<<"Training data size is"<<training_data_size<<endl;
    data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
	double data_log_likelihood=0;	
	Matrix<double,Dynamic,Dynamic> current_c_for_gradCheck, current_h_for_gradCheck, current_c,current_h, init_c, init_h;

	//init_c.setZero(myParam.num_hidden,minibatch_size);
	//init_h.setZero(myParam.num_hidden,minibatch_size);
	//c_last.setZero(numParam.num_hidden, minibatch_size);
	//h_last.setZero(numParam.num_hidden, minibatch_size);
	
    for(data_size_t batch=0;batch<num_batches;batch++)
    {
			current_c.setZero(myParam.num_hidden, minibatch_size);
			current_h.setZero(myParam.num_hidden, minibatch_size);			
            if (batch > 0 && batch % 100 == 0)
            {
	        cerr << batch <<"...";
            } 

            data_size_t minibatch_start_index = minibatch_size * batch;
			data_size_t minibatch_end_index = min(training_data_size-1, static_cast<data_size_t> (minibatch_start_index+minibatch_size-1));
			//cerr<<"Minibatch start index is "<<minibatch_start_index<<endl;
			//cerr<<"Minibatch end index is "<<minibatch_end_index<<endl;

      	  int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), training_data_size - minibatch_start_index);
	  	  //cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
	  
		  /*
	      //ALTERNATIVE OPTION IF YOU'RE NOT USING eigen map interface on the mmapped file
		    Matrix<int,Dynamic,Dynamic> minibatch;// = training_data.middleCols(minibatch_start_index, current_minibatch_size);
			//cerr<<"Minibatch start index "<<minibatch_start_index<<endl;
			//cerr<<"Minibatch size "<<current_minibatch_size<<endl;
	            if (use_mmap_file == true) {
	            minibatch.setZero(ngram_size,current_minibatch_size);
	            //now reading the ngrams from the mmaped file
	              for (int k=0; k<ngram_size; k++){
	                for (data_size_t index = 0 ; index<current_minibatch_size; index++) {
					  data_size_t current_index = index + minibatch_start_index;
					  //cerr<<"the value in the mmap file "<<index<<" "<<k<<" is "<<training_data_flat_mmap->at(current_index*ngram_size+k)<<endl;
	                  minibatch(k,index) = training_data_flat_mmap->at(current_index*ngram_size+k);
	                }
	              }
	            } else {
	              minibatch = training_data.middleCols(minibatch_start_index, current_minibatch_size);
	            }
		  	*/
	  
	  

		  	//double adjusted_learning_rate = current_learning_rate;
			//cerr<<"Adjusted learning rate is"<<adjusted_learning_rate<<endl;
            //cerr<<"Adjusted learning rate: "<<adjusted_learning_rate<<endl;

            ///// Forward propagation

            //prop.fProp(minibatch.topRows(ngram_size-1));
		//
			//Taking the input and output sentence and setting the training data to it.
			//Getting a minibatch of sentences
			vector<int> minibatch_input_sentences, minibatch_output_sentences, minibatch_sequence_cont_sentences;
			unsigned int max_input_sent_len,max_output_sent_len;
			unsigned int minibatch_output_tokens,minibatch_input_tokens, minibatch_sequence_cont_tokens;
			minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;
			miniBatchify(training_input_sent, 
							minibatch_input_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_input_sent_len,
							1,
							minibatch_input_tokens);
			miniBatchify(training_output_sent, 
							minibatch_output_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_output_sent_len,
							0,
							minibatch_output_tokens);	
			//cerr<<"Max output sent len is "<<max_output_sent_len<<endl;
			//getchar();
							/*				
			miniBatchify(training_sequence_cont_sent, 
							minibatch_sequence_cont_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_sent_len,
							1,
							minibatch_sequence_cont_tokens);	
							*/	
			/*
			training_input_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(training_input_sent[batch].data(), 
											training_input_sent[batch].size(),
											current_minibatch_size);
											
			training_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(training_output_sent[batch].data(),
																			training_output_sent[batch].size(),
																			current_minibatch_size);
			*/
			training_input_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_input_sentences.data(), 
											max_input_sent_len,
											current_minibatch_size);
							
			training_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_output_sentences.data(),
																			max_output_sent_len,
																			current_minibatch_size);
			//training_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_sequence_cont_sentences.data(),
			//																max_sent_len,
			//																current_minibatch_size);
			training_sequence_cont_sent_data = Array<int,Dynamic,Dynamic>	();
			//cerr<<"training_input_sent_data "<<training_input_sent_data<<endl;
			//cerr<<"training_output_sent_data"<<training_output_sent_data<<endl;
			//exit(0);
			//Calling fProp. Note that it should not matter for fProp if we're doing log 
			//or NCE loss													
			if (myParam.gradient_check) {
				current_c_for_gradCheck = current_c;
				current_h_for_gradCheck = current_h;
				cerr<<"current_c_for_gradCheck "<<current_c_for_gradCheck<<endl;
				cerr<<"current_h_for_gradCheck "<<current_h_for_gradCheck<<endl;
			}													
			init_c = current_c;
			init_h = current_h; 			
			prop.fProp(training_input_sent_data,
					training_output_sent_data,
						0,
						max_input_sent_len-1,
						current_c,
						current_h,
						training_sequence_cont_sent_data);	
						
		  	double adjusted_learning_rate = current_learning_rate;
			if (!myParam.norm_clipping){
				adjusted_learning_rate /= current_minibatch_size;			
			}
		    //if (loss_function == NCELoss)
		    //{


		    //}
		    //else if (loss_function == LogLoss)
		    //{
				
				//computing losses
			    prop.computeLosses(training_output_sent_data,
					 data_log_likelihood,
					 myParam.gradient_check,
					 myParam.norm_clipping,
					 loss_function,
					 unigram,
					 num_noise_samples,
					 rng,
					 softmax_nce_loss); //, 
				//Calling backprop
			    prop.bProp(training_input_sent_data,
					training_output_sent_data,
					 myParam.gradient_check,
					 myParam.norm_clipping); //, 
					 //init_c,
					 //init_h,
					 //training_sequence_cont_sent_data); 	

	 			//Checking the compute probs function
	 			//prop.computeProbs(training_output_sent_data,
	 			//					data_log_likelihood);	
				//cerr<<"training_input_sent_data len"		
				if (myParam.gradient_check) {		
					cerr<<"Checking gradient"<<endl;
						 
					prop.gradientCheck(training_input_sent_data,
						 		 training_output_sent_data,
								 current_c_for_gradCheck,
								 current_h_for_gradCheck,
								 unigram,
								 num_noise_samples,
					   			 rng_grad_check,
					   			 loss_function,
								 softmax_nce_loss,
								 training_sequence_cont_sent_data);
					//for the next minibatch, we want the range to be updated as well
					rng_grad_check = rng;
				}
				//getchar();											 
				//Updating the gradients
				prop.updateParams(adjusted_learning_rate,
							//max_sent_len,
							max_input_sent_len,
					  		current_momentum,
							myParam.L2_reg,
							myParam.norm_clipping,
							myParam.norm_threshold,
							loss_function);														
	
				//Resetting the gradients

				prop.resetGradient();
	      //}
		  
	 }
	 cerr << "done." << endl;
	if (loss_function == LogLoss)
	{
		//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
		//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
		//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
		//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
		//log_likelihood /= sent_len;		
	    cerr << "Training log-likelihood base e:      " << data_log_likelihood << endl;
		cerr << "Training log-likelihood base 2:     " << data_log_likelihood/log(2.) << endl;
		cerr << "Training cross entropy in base 2 is "<<data_log_likelihood/(log(2.)*total_output_tokens)<< endl;
		cerr << "         perplexity:                 "<< exp(-data_log_likelihood/total_output_tokens) << endl;
	}
	else if (loss_function == NCELoss) 
	{
	    cerr << "Training NCE log-likelihood: " << data_log_likelihood << endl;
		cerr << "Average NCE log-likelihood " << data_log_likelihood/(total_output_tokens*myParam.num_noise_samples) << endl;
	}
	
	if (myParam.use_momentum)
        current_momentum += momentum_delta;

	#ifdef USE_CHRONO
	cerr << "Propagation times:";
	for (int i=0; i<timer.size(); i++)
	  cerr << " " << timer.get(i);
	cerr << endl;
	#endif
	
	
	if (myParam.model_prefix != "")
	{
	    cerr << "Writing model" << endl;
	    if (myParam.input_words_file != "") {
	        nn.write(myParam.model_prefix + ".encoder." + lexical_cast<string>(epoch+1), input_words, output_words);
			nn_decoder.write(myParam.model_prefix + ".decoder." + lexical_cast<string>(epoch+1), output_words, output_words);	
		}
	    else
	        nn.write(myParam.model_prefix + "." + lexical_cast<string>(epoch+1));
	}
	
	
		
        if (epoch % 1 == 0 && validation_data_size > 0)
        {
			cerr<<"Computing validation perplexity..."<<endl;
            //////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
            ////////////////////////////////////////////////////////////////

            double log_likelihood = 0.0;
			
		    //Matrix<double,Dynamic,Dynamic> scores(output_vocab_size, validation_minibatch_size);
		    //Matrix<double,Dynamic,Dynamic> output_probs(output_vocab_size, validation_minibatch_size);
		    //Matrix<int,Dynamic,Dynamic> minibatch(ngram_size, validation_minibatch_size);
			Matrix<double,Dynamic,Dynamic> current_validation_c,current_validation_h;
			current_validation_c.setZero(myParam.num_hidden, validation_minibatch_size);
			current_validation_h.setZero(myParam.num_hidden, validation_minibatch_size);
			
            for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
            {
				current_validation_c.setZero(myParam.num_hidden, validation_minibatch_size);
				current_validation_h.setZero(myParam.num_hidden, validation_minibatch_size);
				double minibatch_log_likelihood = 0.;
	            data_size_t minibatch_start_index = validation_minibatch_size * validation_batch;
				data_size_t minibatch_end_index = min(validation_data_size-1, static_cast<data_size_t> (minibatch_start_index+validation_minibatch_size-1));
				//cerr<<"Minibatch start index is "<<minibatch_start_index<<endl;
				//cerr<<"Minibatch end index is "<<minibatch_end_index<<endl;

	      	  	int current_minibatch_size = min(static_cast<data_size_t>(validation_minibatch_size), validation_data_size - minibatch_start_index);
		  	  	//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
 
  
				//Taking the input and output sentence and setting the validation data to it.
				//Getting a minibatch of sentences
				vector<int> minibatch_input_sentences, minibatch_output_sentences, minibatch_sequence_cont_sentences;
				unsigned int max_input_sent_len,max_output_sent_len;
				unsigned int minibatch_output_tokens,minibatch_input_tokens,minibatch_sequence_cont_tokens;
				minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;				
				miniBatchify(validation_input_sent, 
								minibatch_input_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_input_sent_len,
								1,
								minibatch_input_tokens);
				miniBatchify(validation_output_sent, 
								minibatch_output_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								0,
								minibatch_output_tokens);	
			/*											
				miniBatchify(validation_sequence_cont_sent, 
								minibatch_sequence_cont_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_sent_len,
								1,
								minibatch_sequence_cont_tokens);
			*/					
				validation_input_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_input_sentences.data(), 
												max_input_sent_len,
												current_minibatch_size);
						
				validation_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_output_sentences.data(),
																				max_output_sent_len,
																				current_minibatch_size);
				//validation_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_sequence_cont_sentences.data(),
				//																	max_sent_len,
				//																	current_minibatch_size);
			    validation_sequence_cont_sent_data = Array<int,Dynamic,Dynamic>	()	;																
				//cerr<<"Validation input sent data "<<validation_input_sent_data<<endl;
				//cerr<<"Validation output sent data "<<validation_output_sent_data<<endl;
																			
				//cerr<<"training_input_sent_data "<<training_input_sent_data<<endl;
				//cerr<<"training_output_sent_data"<<training_output_sent_data<<endl;
				//exit(0);
				//Calling fProp. Note that it should not matter for fProp if we're doing log 
				//or NCE loss															
				prop_validation.fProp(validation_input_sent_data,
										validation_output_sent_data,
										0,
										max_input_sent_len-1, 
										current_validation_c, 
										current_validation_h,
										validation_sequence_cont_sent_data);	
		
	
						 
		 		prop_validation.computeProbsLog(validation_output_sent_data,
		 		 			  	minibatch_log_likelihood);
				//cerr<<"Minibatch log likelihood is "<<minibatch_log_likelihood<<endl;
				log_likelihood += minibatch_log_likelihood;
			}
				
	        //cerr << "Validation log-likelihood: "<< log_likelihood << endl;
	        //cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;
		    cerr << "		Validation log-likelihood base e:      " << log_likelihood << endl;
			cerr << "		Validation log-likelihood base 2:     " << log_likelihood/log(2.) << endl;
			cerr<<  "		Validation cross entropy in base 2 is "<< log_likelihood/(log(2.)*total_validation_output_tokens)<< endl;
			cerr << "         		perplexity:                    "<< exp(-log_likelihood/total_validation_output_tokens) << endl;
			
		    // If the validation perplexity decreases, halve the learning rate.
	        //if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA")
			if (epoch > 0 && 1.002*log_likelihood < current_validation_ll && myParam.parameter_update != "ADA") //This is what mikolov does 
	        { 
	            current_learning_rate /= 2;
	        }
	        current_validation_ll = log_likelihood;
	 
		}
	
    }
    return 0;
}

