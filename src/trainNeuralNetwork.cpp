//#define EIGEN_NO_MALLOC
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
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_mapped_file.hpp>
#include <boost/interprocess/containers/vector.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "maybe_omp.h"
#include <tclap/CmdLine.h>

//#include "fastonebigheader.h"
#include "define.h"
//#include "constants.h"
#include "model.h"
#include "propagator.h"
#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "multinomial.h"
#include "vocabulary.h"

//#include "gradientCheck.h"

//#define EIGEN_DONT_PARALLELIZE


using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost;
using namespace boost::random;

using namespace nplm;

namespace ip = boost::interprocess;
typedef unordered_map<Matrix<int,Dynamic,1>, precision_type> vector_map;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;


//typedef long long int data_size_t; // training data can easily exceed 2G instances

int main(int argc, char** argv)
{ 
	/*
	if (!argc>=2)
	{
	    cerr << "ERROR. You need to supply necessary files. Please type trainNeuralNetwork --help."<<endl;
	    return 1;
	}	
	*/
	cerr<<"precision type is "<<sizeof(precision_type)<<endl;
	srand (time(NULL));
	cerr<<setprecision(16);
    ios::sync_with_stdio(false);
    bool use_mmap_file, randomize=0, arg_run_lm = 0, arg_carry_states=0, arg_run_tagger=0;
    param myParam;
	int arg_seed;
    try {
      // program options //
      CmdLine cmd("Trains a LSTM encoder decoder or a language model.", ' ' , "0.1\n","");

      // The options are printed in reverse order

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);


      ValueArg<int> num_noise_samples("", "num_noise_samples", "Number of noise samples for noise-contrastive estimation. Default: 100.", false, 100, "int", cmd);

      ValueArg<precision_type> L2_reg("", "L2_reg", "L2 regularization strength (hidden layer weights only). Default: 0.", false, 0.0, "precision_type", cmd);

      ValueArg<precision_type> learning_rate("", "learning_rate", "Learning rate for stochastic gradient ascent. Default: 1.", false, 1., "precision_type", cmd);
	  ValueArg<precision_type> fixed_partition_function("", "fixed_partition_function", "Fixed log normalization constant value. Default: 0.", false, 0., "precision_type", cmd);


      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 128.", false, 128, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 128.", false, 128, "int", cmd);

      ValueArg<int> num_epochs("", "num_epochs", "Number of epochs. Default: 10.", false, 10, "int", cmd);

      ValueArg<precision_type> init_range("", "init_range", "Maximum (of uniform) or standard deviation (of normal) for initialization. Default: 0.1", false, 0.1, "precision_type", cmd);
	  ValueArg<precision_type> init_forget("", "init_forget", "value to initialize the bias of the forget gate. Default: 0", false, 0, "precision_type", cmd);
      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters from a normal distribution. 1 = normal, 0 = uniform. Default: 0. \
		  (initialize from a uniform distribution)", false, 0, "bool", cmd);
      ValueArg<int> seed("", "seed", "The seed for the random number generator (used for initializing the model parameters). \
		   Default: 1234.", false, 1234, "int", cmd);

	  ValueArg<int> input_embedding_dimension("", "input_embedding_dimension", "Number of input embedding dimensions. \
		  Default: Same as num_hidden.", false, 0, "int", cmd);
      ValueArg<string> loss_function("", "loss_function", "Loss function (log, nce). Default: log.", false, "log", "string", cmd);
      //ValueArg<string> activation_function("", "activation_function", "Activation function (identity, rectifier, tanh, hardtanh). Default: rectifier.", false, "rectifier", "string", cmd);
      ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 64. All gates, cells, hidden layers, \n \
		  							input and output embedding dimension are set to this value", false, 64, "int", cmd);

      ValueArg<string> model_prefix("", "model_prefix", "Prefix for output model files." , true, "", "string", cmd);

      ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);
      ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);
	  //ValueArg<string> input_sent_file("", "input_sent_file", "Input sentences file." , true, "", "string", cmd);
	  //ValueArg<string> output_sent_file("", "output_sent_file", "Input sentences file." , true, "", "string", cmd);
	  ValueArg<string> training_sent_file("", "training_sent_file", "Training sentences file" , true, "", "string", cmd);
	  ValueArg<string> validation_sent_file("", "validation_sent_file", "Validation sentences file" , true, "", "string", cmd);
	  
	  ValueArg<bool> gradient_check("", "gradient_check", "Do you want to do a gradient check or not. 1 = Yes, 0 = No. Default: 0.", false, 0, "bool", cmd);
	  
      //ValueArg<string> train_file("", "train_file", "Training data (one numberized example per line)." , true, "", "string", cmd);
	  ValueArg<bool> norm_clipping("", "norm_clipping", "Do you want to do norm clipping or gradient clipping. 1 = norm cilpping, \n \
		  			0 = gradient clipping. Default: 1.", false, 1, "bool", cmd);
	  ValueArg<bool> run_lm("", "run_lm", "Run as a language model, \n \
		  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);	
	  ValueArg<bool> run_tagger("", "run_tagger", "Run as a tagger, \n \
	  	  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);		  
	  ValueArg<bool> carry_states("", "carry_states", "Carry the hidden states from one minibatch to another. This option is for \n \
		  			language models only. Carrying hidden states over can improve perplexity. \n \
						1 = yes. Default: 0 (Do not carry hidden states).", false, 0, "bool", cmd);		  
	  ValueArg<bool> reverse_input("", "reverse", "Reverse the input sentence before training, \n \
		  			1 = yes. Default: 0 (No reversing).", false, 0, "bool", cmd);		    
	  //ValueArg<bool> restart_states("", "restart_states", "If yes, then the hidden and cell values will be restarted after every minibatch \n \
	//	  Default: 1 = yes, \n \
	//	  			0 = gradient clipping. Default: 0.", false, 0, "bool", cmd);	  
      //ValueArg<string> model_file("", "model_file", "Model file.", false, "", "string", cmd);
	  ValueArg<precision_type> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "precision_type", cmd);
	  ValueArg<precision_type> dropout_probability("", "dropout_probability", "Dropout probability. Default 0: No dropout", false,0., "precision_type", cmd);
	  
      ValueArg<int> max_epoch("", "max_epoch", "After max_epoch, the learning rate is halved for every subsequent epoch. \n \
		  If not supplied, then the learning rate is modified based on the valdation set. Default: -1", false, -1, "int", cmd);	
	  ValueArg<string> input_embeddings_file("","input_embeddings_file", "Read the input embeddings from the specified file. \n \
		  Default: none", false,"","string",cmd);  
      cmd.parse(argc, argv);



	  myParam.training_sent_file = training_sent_file.getValue();
	  myParam.validation_sent_file = validation_sent_file.getValue();
	  myParam.reverse_input = reverse_input.getValue();

      myParam.model_prefix = model_prefix.getValue();

	  /*
      if (vocab_size.getValue() >= 0) {
	      myParam.input_vocab_size = myParam.output_vocab_size = vocab_size.getValue();
      }
	  */
      myParam.num_hidden = num_hidden.getValue();
      //myParam.activation_function = activation_function.getValue();
      myParam.loss_function = loss_function.getValue();

      myParam.num_threads = num_threads.getValue();

      myParam.num_noise_samples = num_noise_samples.getValue();


      myParam.minibatch_size = minibatch_size.getValue();
	  //myParam.minibatch_size = 1;
      myParam.validation_minibatch_size = validation_minibatch_size.getValue();
	  //myParam.validation_minibatch_size =1;
      myParam.num_epochs= num_epochs.getValue();
      myParam.learning_rate = learning_rate.getValue();

	  myParam.adagrad_epsilon = 0;
	  myParam.fixed_partition_function = fixed_partition_function.getValue();
      //myParam.use_momentum = use_momentum.getValue();
	  myParam.use_momentum = 0;

      myParam.L2_reg = L2_reg.getValue();
      myParam.init_normal= init_normal.getValue();
      myParam.init_range = init_range.getValue();
	  myParam.init_forget = init_forget.getValue();
	  myParam.max_epoch = max_epoch.getValue();
      //myParam.normalization_init = normalization_init.getValue();
      //myParam.parameter_update = parameter_update.getValue();
	  myParam.parameter_update = "SGD";
	  myParam.gradient_check = gradient_check.getValue();
	  myParam.norm_clipping = norm_clipping.getValue();
	  myParam.norm_threshold = norm_threshold.getValue();
	  myParam.dropout_probability = dropout_probability.getValue();
	  myParam.input_embedding_dimension = input_embedding_dimension.getValue();
	  myParam.input_embeddings_file = input_embeddings_file.getValue();
	  //myParam.restart_states = norm_threshold.getValue();
	  
	  //arg_run_lm = run_lm.getValue();
	  arg_run_lm = 1;
	  arg_run_tagger = run_tagger.getValue();
	  arg_seed = seed.getValue();
	  arg_carry_states = carry_states.getValue();
	  if (arg_run_lm == 0 && arg_carry_states == 1){
		  cerr<<"--carry_states 1 can only be used with --run_lm 1"<<endl;
		  exit(1);
	  }
	  //myParam.load_encoder_file = load_encoder_file.getVale();
	  //myParam.load_decoder_file = load_decoder_file.getVale();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
      //cerr << train_file.getDescription() << sep << train_file.getValue() << endl;
      //cerr << validation_file.getDescription() << sep << validation_file.getValue() << endl;
	  cerr << training_sent_file.getDescription() << sep << training_sent_file.getValue() << endl;
	  cerr << validation_sent_file.getDescription() << sep << validation_sent_file.getValue() << endl;
      cerr << model_prefix.getDescription() << sep << model_prefix.getValue() << endl;

	  cerr << norm_clipping.getDescription() << sep << norm_clipping.getValue() <<endl;
	  cerr << norm_threshold.getDescription() << sep << norm_threshold.getValue() <<endl;
	  cerr << dropout_probability.getDescription() << sep << dropout_probability.getValue() <<endl;
	  cerr << gradient_check.getDescription() <<sep <<gradient_check.getValue() <<endl;
	  cerr << max_epoch.getDescription() << sep << max_epoch.getValue() <<endl;
	  //cerr << restart_states.getDescription() <<sep <<restart_states.getValue() <<endl;
	  cerr << run_lm.getDescription() <<sep <<run_lm.getValue() <<endl;
	  cerr << carry_states.getDescription() <<sep <<carry_states.getValue() <<endl;
	  cerr << reverse_input.getDescription() <<sep <<reverse_input.getValue() <<endl;
	  cerr << seed.getDescription() << sep << seed.getValue() <<endl;
	  cerr << loss_function.getDescription() << sep << loss_function.getValue() << endl;
	  cerr << num_noise_samples.getDescription() << sep << num_noise_samples.getValue() << endl;
	  cerr << fixed_partition_function.getDescription() << sep << fixed_partition_function.getValue() << endl;
	  cerr << input_embedding_dimension.getDescription() << sep << input_embedding_dimension.getValue() << endl;
	  cerr << input_embeddings_file.getDescription() << sep << input_embeddings_file.getValue() << endl;
	  //cerr << load_encoder_file.getDescription() <<sep <<load_encoder_file.getValue() <<endl;
	  //cerr << load_decoder_file.getDescription() <<sep <<load_decoder_file.getValue() <<endl;
	  if (arg_run_lm == 1) {
		  cerr<<"Running as a LSTM language model"<<endl;
	  } else {
		  cerr<<"Running as a LSTM sequence to sequence model"<<endl;
	  }
	  //exit(0);
	  //cerr << fixed_partition_function.getDescription() <<sep <<fixed_partition_function.getValue() <<endl;
	  

      cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;

	  
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
	  
      cerr << num_threads.getDescription() << sep << num_threads.getValue() << endl;
	  

    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    myParam.num_threads = setup_threads(myParam.num_threads);
    int save_threads;

    //unsigned seed = std::time(0);
    //unsigned seed = 1234; //for testing only

    mt19937 rng(arg_seed), rng_grad_check(arg_seed);

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read training data

    vector<int> training_data_flat;
	vector< vector<int> > training_input_sent, validation_input_sent, training_sequence_cont_sent;
	vector< vector<int> > training_output_sent, validation_output_sent, validation_sequence_cont_sent;
	vector< vector<int> > decoder_training_input_sent, decoder_training_output_sent;
	vector <vector<int> > decoder_validation_input_sent, decoder_validation_output_sent;
	vector< vector<string> > word_training_input_sent, word_validation_input_sent;
	vector< vector<string> > word_training_output_sent, word_validation_output_sent;

    // Construct vocabulary
    vocabulary input_vocab, output_vocab;
	vocabulary decoder_input_vocab, decoder_output_vocab;
    int start, stop;

    vec * training_data_flat_mmap;
    data_size_t training_data_size, validation_data_size; //num_tokens;
    ip::managed_mapped_file mmap_file;

	
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


	//Even sentences are input
	//if (arg_run_lm == 0) {
	//do not add start stop for now.
	readEvenSentFile(myParam.training_sent_file, 
		word_training_input_sent, 
		total_input_tokens, 
		0, 
		0, 
		myParam.reverse_input);

	
	//Reading output 
	readOddSentFile(myParam.training_sent_file, 
		word_training_output_sent, 
		total_output_tokens,
		0,
		1, 
		0); //We do not 	everse the output. We do not add start stop
				
	int decoder_input_vocab_size;
	int decoder_output_vocab_size;
	decoder_input_vocab_size = decoder_output_vocab_size = 0;
	//If load input model and load output model have been specified
	//After reading the sentence file, create the input and output vocabulary if it hasn't already been specified

	buildDecoderVocab(word_training_input_sent, 
							decoder_input_vocab,
							0,
							0);
	decoder_input_vocab_size = decoder_input_vocab.size();
	
	buildDecoderVocab(word_training_output_sent, 
							decoder_output_vocab,
							0,
							0);					
	decoder_output_vocab_size = decoder_output_vocab.size();
	myParam.output_vocab_size = decoder_output_vocab.size();
	myParam.input_vocab_size = decoder_input_vocab.size();
	
	/*
	//cerr<<"Output vocab size is "<<myParam.output_vocab_size<<endl;	
	cerr<<"Decoder input vocab size is "<<decoder_input_vocab_size<<endl;
	cerr<<"Decoder output vocab size is "<<decoder_output_vocab_size<<endl;
	cerr<<"myParam.output_vocab_size "<<myParam.output_vocab_size<<endl;
	cerr<<"myParam.input_vocab_size "<<myParam.input_vocab_size<<endl;
	
	cerr<<"Decoder input vocab is "<<endl;
	decoder_input_vocab.print_vocabulary();
	cerr<<"Decoder output vocab is "<<endl;
	decoder_output_vocab.print_vocabulary();
	getchar();
	*/
	
	//Creating separate decoder input vocab and decoder output vocab
	
   	integerize(word_training_output_sent, 
   			   decoder_training_output_sent,
   			   decoder_output_vocab);
  	integerize(word_training_input_sent, 
  			   decoder_training_input_sent,
  			   decoder_input_vocab);				   	
	assert(decoder_training_output_sent.size() == decoder_training_input_sent.size());   						

	
	training_data_size = decoder_training_output_sent.size();

	data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;


	cerr<<"Number of output tokens "<<total_output_tokens<<endl;
	cerr<<"Number of minibatches "<<num_batches<<endl;
    //data_size_t training_data_size = num_tokens / myParam.ngram_size;
	
    cerr << "Number of training instances "<< decoder_training_output_sent.size() << endl;
    cerr << "	 "<< decoder_validation_output_sent.size() << endl;
	
    Matrix<int,Dynamic,Dynamic> training_data;
	Matrix<int,Dynamic,Dynamic> training_input_sent_data, training_output_sent_data;
	Matrix<int,Dynamic,Dynamic> decoder_training_input_sent_data, decoder_training_output_sent_data;
	Matrix<int,Dynamic,Dynamic> decoder_validation_input_sent_data, decoder_validation_output_sent_data;
	Matrix<int,Dynamic,Dynamic> validation_input_sent_data, validation_output_sent_data;
	Array<int,Dynamic,Dynamic> validation_input_sequence_cont_sent_data, training_input_sequence_cont_sent_data;
	Array<int,Dynamic,Dynamic> validation_output_sequence_cont_sent_data, training_output_sequence_cont_sent_data;
    //(training_data_flat.data(), myParam.ngram_size, training_data_size);
    
	
    // Read validation data
    //vector<int> validation_data_flat;
    validation_data_size = 0;
    

	//cerr<<"printing input vocabulary"<<endl;
	//input_vocab.print_vocabulary();
	//cerr<<"printing output vocabulary"<<endl;
	//output_vocab.print_vocabulary();
	//getchar();
	
	//if (myParam.input_validation_sent_file != ""){
	if (myParam.validation_sent_file != "") {
		

		readEvenSentFile(myParam.validation_sent_file, 
					word_validation_input_sent, 
					total_validation_input_tokens,
					0,
					0,
					myParam.reverse_input);

		cerr<<"Validation input tokens "<<total_validation_input_tokens<<endl;									

		readOddSentFile(myParam.validation_sent_file, 
					word_validation_output_sent, 
					total_validation_output_tokens,
					0,
					0,
					0);

		integerize(word_validation_output_sent, 
				   decoder_validation_output_sent,
				   decoder_output_vocab);	

		integerize(word_validation_input_sent, 
				   decoder_validation_input_sent, 
				   decoder_input_vocab);	
				      						
		cerr<<"Validation output tokens "<<total_validation_output_tokens<<endl;									
		
	}
	

	validation_data_size = word_validation_output_sent.size();
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
	


    ///// Create and initialize the neural network and associated propagators.
	if (myParam.input_embedding_dimension == 0) {
		cerr<<"Setting input embedding dimension to num_hidden"<<endl;
		myParam.input_embedding_dimension = myParam.num_hidden;
	} 
	myParam.output_embedding_dimension = myParam.num_hidden;
	
    model nn;
    nn.resize(myParam.ngram_size,
	        decoder_input_vocab_size,
	        decoder_output_vocab_size,
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
	input.resize(decoder_input_vocab_size,
		    myParam.input_embedding_dimension,
		    myParam.num_hidden);

	input.initialize(rng,
        myParam.init_normal,
        myParam.init_range,
        myParam.parameter_update,
        myParam.adagrad_epsilon);
	
	nn.set_input(input);
				
	//getchar();
	//cerr<<"printing decoder stuff"<<endl;
	model nn_decoder;
	    nn_decoder.resize(myParam.ngram_size,
	        decoder_input_vocab_size,
	        decoder_output_vocab_size,
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
					decoder_input_vocab_size,
					myParam.input_embedding_dimension);
		cerr<<"decoder_input_vocab_size "<<decoder_input_vocab_size<<endl;
		decoder_input.resize(decoder_input_vocab_size,
		    myParam.input_embedding_dimension,
		    myParam.num_hidden);

		decoder_input.initialize(rng,
	        myParam.init_normal,
	        myParam.init_range,
	        myParam.parameter_update,
	        myParam.adagrad_epsilon);
		nn_decoder.set_input(decoder_input);	
		//Reading embeddings if they have been defined 
		if (myParam.input_embeddings_file != ""){
			decoder_input.readEmbeddingsFromFile(myParam.input_embeddings_file,
				decoder_input_vocab);
			input.readEmbeddingsFromFile(myParam.input_embeddings_file,
				decoder_input_vocab);				
		}
		
	//getchar();	
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL


	
    if (myParam.model_file != ""){
      nn.read(myParam.model_file);
      cerr<<"reading the model"<<endl;
    } else {
      //nn.set_activation_function(string_to_activation_function(myParam.activation_function));
	  rng_grad_check = rng; //The range for gradient check should have exactly the same state as rng for the NCE gradient checking to work
	  
    }
    loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

    propagator<Google_input_node, google_input_model> prop(nn, 
														nn_decoder, 
														myParam.minibatch_size);
	if (myParam.dropout_probability > 0 ){
		prop.resizeDropout(myParam.minibatch_size, myParam.dropout_probability);
	}

    propagator<Google_input_node, google_input_model> prop_validation(nn, 
															nn_decoder, 
															myParam.validation_minibatch_size);
	vector<data_size_t> unigram_counts;
	multinomial<data_size_t> unigram;																
	if (loss_function == NCELoss){
	    vector<data_size_t> unigram_counts = vector<data_size_t>(decoder_output_vocab_size,0);
		//for (int index=0; index<decoder_output_vocab_size; index++){
		//	unigram_counts[index] = 1; //Currently using uniform noise
		//}
		for (int sent_index=0; sent_index<decoder_training_output_sent.size(); sent_index++){
			for (int word_index=0; word_index<decoder_training_output_sent[sent_index].size(); word_index++){
				unigram_counts[decoder_training_output_sent[sent_index][word_index]] += 1;
			}
		}
		unigram = multinomial<data_size_t> (unigram_counts);
	}


	//IF we're using NCE, then the minibatches have different sizes
	if (loss_function == NCELoss)
		prop.resizeNCE(myParam.num_noise_samples, myParam.fixed_partition_function, unigram);
																
	cerr<<"Finished reading the data"<<endl;
	//exit(1);
    
    ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////


    cerr<<"Number of training minibatches: "<<num_batches<<endl;
	
	
    int num_validation_batches = 0;
    if (validation_data_size > 0)
    {
        num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
		cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
    } 
	
    precision_type current_momentum = myParam.initial_momentum;
    precision_type momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
    precision_type current_learning_rate = myParam.learning_rate;
    precision_type current_validation_ll = 0.0;
	precision_type current_accuracy = 0.;

    int ngram_size = myParam.ngram_size;
    int input_vocab_size = myParam.input_vocab_size;
    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
    int validation_minibatch_size = myParam.validation_minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;

	
	precision_type best_perplexity = 999999999;
	precision_type best_accuracy = 0.;
	int best_model = 0;	
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

		precision_type log_likelihood = 0.0;

		int num_samples = 0;
		if (loss_function == LogLoss)
		    num_samples = output_vocab_size;
		else if (loss_function == NCELoss)
		    num_samples = 1+num_noise_samples;
		//Generating 10 samples
	
	
		//cerr<<"Training data size is"<<training_data_size<<endl;
	    data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
		precision_type data_log_likelihood=0;	
		Matrix<precision_type,Dynamic,Dynamic> current_c_for_gradCheck, current_h_for_gradCheck, current_c,current_h, init_c, init_h;

		init_c.setZero(myParam.num_hidden,minibatch_size);
		init_h.setZero(myParam.num_hidden,minibatch_size);
		current_c.setZero(myParam.num_hidden, minibatch_size);
		current_h.setZero(myParam.num_hidden, minibatch_size);			
		//c_last.setZero(numParam.num_hidden, minibatch_size);
		//h_last.setZero(numParam.num_hidden, minibatch_size);
	
		//cerr<<"About to start training "<<endl;
    for(data_size_t batch=0;batch<num_batches;batch++)
    {
			//err<<"batch is "<<batch<<endl;
			if (arg_carry_states == 0) {
				current_c.setZero(myParam.num_hidden, minibatch_size);
				current_h.setZero(myParam.num_hidden, minibatch_size);			
			}
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
	  


		  	//precision_type adjusted_learning_rate = current_learning_rate;
			//cerr<<"Adjusted learning rate is"<<adjusted_learning_rate<<endl;
            //cerr<<"Adjusted learning rate: "<<adjusted_learning_rate<<endl;

            ///// Forward propagation

            //prop.fProp(minibatch.topRows(ngram_size-1));
		//
			//Taking the input and output sentence and setting the training data to it.
			//Getting a minibatch of sentences
		  	
			vector<int> minibatch_input_sentences, minibatch_output_sentences, minibatch_sequence_cont_sentences;
			vector<int> minibatch_input_sequence_cont_sentences, minibatch_output_sequence_cont_sentences;
			vector<int> minibatch_decoder_output_sentences, minibatch_decoder_input_sentences;
			unsigned int max_input_sent_len, max_output_sent_len;
			unsigned int minibatch_output_tokens,minibatch_input_tokens, minibatch_sequence_cont_tokens;
			minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;
			max_input_sent_len = max_output_sent_len = 0;
			//cerr<<"reading data"<<endl;
			

			miniBatchifyDecoder(decoder_training_output_sent, 
							minibatch_decoder_output_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_output_sent_len,
							minibatch_output_tokens,
							1,
							-1);		
			minibatch_output_tokens =0;						
			miniBatchifyDecoder(decoder_training_input_sent, 
							minibatch_decoder_input_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_output_sent_len,
							minibatch_output_tokens,
							1,
							0);									
			decoder_training_output_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_output_sentences.data(),
																						max_output_sent_len,
																						current_minibatch_size);
			decoder_training_input_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_input_sentences.data(),
																						max_output_sent_len,
																						current_minibatch_size);	

																						
			//cerr<<"decoder_training_input_sent_data "<<decoder_training_input_sent_data<<endl;		
								
			minibatch_output_tokens =0;	
			miniBatchifyDecoder(decoder_training_input_sent, 
							minibatch_output_sequence_cont_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_output_sent_len,
							minibatch_output_tokens,
							0,
							0);	

			training_output_sequence_cont_sent_data = 
					Map< Array<int,Dynamic,Dynamic> >(minibatch_output_sequence_cont_sentences.data(),
																		max_output_sent_len,
																		current_minibatch_size);
			//cerr<<"training_output_sequence_cont_sent_data "<<training_output_sequence_cont_sent_data<<endl;		
																																

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


			if (myParam.dropout_probability > 0.) {
				prop.fPropEncoderDropout(decoder_training_input_sent_data,
					current_c,
					current_h,
					training_output_sequence_cont_sent_data,
					rng);			
					/*
			    prop.fPropDecoderDropout(decoder_training_input_sent_data,
						current_c,
						current_h,
						training_output_sequence_cont_sent_data,
						rng);	
					*/
											
			} else {
				prop.fPropEncoder(decoder_training_input_sent_data,
					current_c,
					current_h,
					training_output_sequence_cont_sent_data);
				//cerr<<"current c before fProp decoder is"<<current_c<<endl;
				//cerr<<"current h before fProp decoder is"<<current_c<<endl;
					/*
			    prop.fPropDecoder(decoder_training_input_sent_data,
						current_c,
						current_h,
						training_output_sequence_cont_sent_data);	
					*/				
			}						
			
								
		  	precision_type adjusted_learning_rate = current_learning_rate;
			if (!myParam.norm_clipping){
				adjusted_learning_rate /= current_minibatch_size;			
			}

				
				//computing losses
				if (myParam.dropout_probability > 0) {
				    prop.computeLossesDropout(decoder_training_output_sent_data,
						 data_log_likelihood,
						 myParam.gradient_check,
						 myParam.norm_clipping,
						 loss_function,
						 unigram,
						 num_noise_samples,
						 rng);
						 //softmax_nce_loss); //, 	
						 /*		
	 				    prop.bPropDecoderDropout(decoder_training_input_sent_data,
	 						 myParam.gradient_check,
	 						 myParam.norm_clipping); //,
						 */
							 
		 			    prop.bPropEncoderDropout(decoder_training_input_sent_data,
		 					 myParam.gradient_check,
		 					 myParam.norm_clipping); 			
							 	
							
				} else {
					//cerr<<" decoder_training_output_sent_data "<<decoder_training_output_sent_data<<endl;
					//cerr<<" decoder_training_input_sent_data "<<decoder_training_input_sent_data<<endl;
					//getchar();
				    prop.computeLosses(decoder_training_output_sent_data,
						 data_log_likelihood,
						 myParam.gradient_check,
						 myParam.norm_clipping,
						 loss_function,
						 unigram,
						 num_noise_samples,
						 rng);
						 //softmax_nce_loss); //,
						 /*
	 				    prop.bPropDecoder(decoder_training_input_sent_data,
	 						 myParam.gradient_check,						  
							 myParam.norm_clipping);
						*/
		 			    prop.bPropEncoder(decoder_training_input_sent_data,
 						 myParam.gradient_check,						  
						 myParam.norm_clipping); 												
				 }	

	
				if (myParam.gradient_check) {		
					//cerr<<"Checking gradient"<<endl;
						 
					prop.gradientCheck(training_input_sent_data,
						 		 decoder_training_input_sent_data,
								 decoder_training_output_sent_data,
								 current_c_for_gradCheck,
								 current_h_for_gradCheck,
								 unigram,
								 num_noise_samples,
					   			 rng_grad_check,
					   			 loss_function,
								 training_input_sequence_cont_sent_data,
								 training_output_sequence_cont_sent_data,
								 arg_run_lm,
								 myParam.dropout_probability);
					//for the next minibatch, we want the range to be updated as well
					rng_grad_check = rng;
				}

				
				//Updaging using local grad norms
				prop.updateParams(adjusted_learning_rate,
							//max_sent_len,
							max_input_sent_len + max_output_sent_len,
					  		current_momentum,
							myParam.L2_reg,
							myParam.norm_clipping,
							myParam.norm_threshold,
							loss_function,
							arg_run_lm);	

				prop.resetGradient();
	      //}
		  
	 }
	 cerr << "done." << endl;
	if (loss_function == LogLoss)
	{
		cerr << "Per symbol training probability      " << exp(data_log_likelihood)/total_output_tokens << endl;
	    cerr << "Training log-likelihood base e:      " << data_log_likelihood << endl;
		cerr << "Training log-likelihood base 2:      " << data_log_likelihood/log(2.) << endl;
		cerr << "Training cross entropy in base 2 is  " <<data_log_likelihood/(log(2.)*total_output_tokens)<< endl;
		cerr << "         perplexity:                 " << exp(-data_log_likelihood/total_output_tokens) << endl;
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
	
		//scale the model before writing it
		//input.scale(1.-myParam.dropout_probability);
		decoder_input.scale(1.-myParam.dropout_probability);
		nn_decoder.scale(1.-myParam.dropout_probability);
		
        if (epoch % 1 == 0 && validation_data_size > 0)
        {
			cerr<<"Computing validation perplexity..."<<endl;
            //////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
            ////////////////////////////////////////////////////////////////

            precision_type log_likelihood = 0.0;
			precision_type correct_validation_labels = 0.;
		    //Matrix<precision_type,Dynamic,Dynamic> scores(output_vocab_size, validation_minibatch_size);
		    //Matrix<precision_type,Dynamic,Dynamic> output_probs(output_vocab_size, validation_minibatch_size);
		    //Matrix<int,Dynamic,Dynamic> minibatch(ngram_size, validation_minibatch_size);
			Matrix<precision_type,Dynamic,Dynamic> current_validation_c,current_validation_h;
			current_validation_c.setZero(myParam.num_hidden, validation_minibatch_size);
			current_validation_h.setZero(myParam.num_hidden, validation_minibatch_size);
			
            for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
            {
				precision_type minibatch_log_likelihood = 0.;
	            data_size_t minibatch_start_index = validation_minibatch_size * validation_batch;
				data_size_t minibatch_end_index = min(validation_data_size-1, static_cast<data_size_t> (minibatch_start_index+validation_minibatch_size-1));
				//cerr<<"Minibatch start index is "<<minibatch_start_index<<endl;
				//cerr<<"Minibatch end index is "<<minibatch_end_index<<endl;

	      	  	int current_minibatch_size = min(static_cast<data_size_t>(validation_minibatch_size), validation_data_size - minibatch_start_index);
		  	  	//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
 
  
				//Taking the input and output sentence and setting the validation data to it.
				//Getting a minibatch of sentences
				vector<int> minibatch_input_sentences, 
					minibatch_output_sentences, 
					minibatch_input_sequence_cont_sentences,
					minibatch_output_sequence_cont_sentences,
					minibatch_decoder_output_sentences,
					minibatch_decoder_input_sentences;
					
				unsigned int max_input_sent_len,max_output_sent_len;
				unsigned int minibatch_output_tokens,minibatch_input_tokens,minibatch_sequence_cont_tokens;
				minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;	
				max_input_sent_len = max_output_sent_len = 0;



				miniBatchifyDecoder(decoder_validation_output_sent, 
								minibatch_decoder_output_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1,
								-1);		

				miniBatchifyDecoder(decoder_validation_input_sent, 
								minibatch_decoder_input_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1,
								0);			
				decoder_validation_output_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_output_sentences.data(),
																							max_output_sent_len,
																							current_minibatch_size);
				decoder_validation_input_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_input_sentences.data(),
																							max_output_sent_len,
																							current_minibatch_size);		
				//cerr<<"decoder_validation_output_sent_data "<<decoder_validation_output_sent_data<<endl;
				//cerr<<"decoder_validation_input_sent_data "<<decoder_validation_input_sent_data<<endl;																								
				miniBatchifyDecoder(decoder_validation_input_sent, 
								minibatch_output_sequence_cont_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								0);		
		

				validation_output_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_output_sequence_cont_sentences.data(),
																			max_output_sent_len,
																			current_minibatch_size);																																																																													

														
				prop_validation.fPropEncoder(decoder_validation_input_sent_data,
						current_validation_c,
						current_validation_h,
						validation_output_sequence_cont_sent_data);	
								
			    prop_validation.fPropDecoder(decoder_validation_input_sent_data,
						current_validation_c,
						current_validation_h,
						validation_output_sequence_cont_sent_data);	
											 
		 		prop_validation.computeProbsLog(decoder_validation_output_sent_data,
		 		 			  	minibatch_log_likelihood,
								correct_validation_labels);
				//cerr<<"Minibatch log likelihood is "<<minibatch_log_likelihood<<endl;
				log_likelihood += minibatch_log_likelihood;
			}
			precision_type accuracy  = correct_validation_labels/total_validation_output_tokens;
	        //cerr << "Validation log-likelihood: "<< log_likelihood << endl;
	        //cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;
			cerr << "		Per symbol validation probability           "<<epoch<<":      " << exp(log_likelihood)/total_validation_output_tokens << endl;
		    cerr << "		Validation log-likelihood base e in epoch   "<<epoch<<":      " << log_likelihood << endl;
			cerr << "		Validation log-likelihood base 2 in epoch   "<<epoch<<":      " << log_likelihood/log(2.) << endl;
			cerr<<  "		Validation cross entropy in base 2 in epoch "<<epoch<<":      "<< log_likelihood/(log(2.)*total_validation_output_tokens)<< endl;
			cerr << "         		perplexity in epoch                 "<<epoch<<":      "<< exp(-log_likelihood/total_validation_output_tokens) << endl;
			cerr << "		Validation accuracy in epoch                "<<epoch<<":      "<< accuracy << endl;
		    // If the validation perplexity decreases, halve the learning rate.
	        //if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA")

			//if (exp(-log_likelihood/total_validation_output_tokens) < best_perplexity){
			if (accuracy < current_accuracy){	
				
				cerr<<"Accuracy on validation improved." <<endl;
				//cerr<<"Previous best accuracy from epoch "<<best_model<<" was "<<<<endl;
				best_perplexity = exp(-log_likelihood/total_validation_output_tokens);
				best_accuracy = accuracy;
				//only write the best model
				if (myParam.model_prefix != "")
				{
				    cerr << "Overwriting the previous best model from epoch " << best_model<< endl;
			        //nn.write(myParam.model_prefix + ".encoder." + lexical_cast<string>(epoch+1), input_vocab.words(), output_vocab.words());
					//n_decoder.write(myParam.model_prefix + ".decoder." + lexical_cast<string>(epoch+1), output_vocab.words(), output_vocab.words());
					nn_decoder.write(myParam.model_prefix + ".decoder.best", decoder_input_vocab.words(), decoder_output_vocab.words());
					if (arg_run_lm == 0) 
						nn.write(myParam.model_prefix + ".encoder.best" , input_vocab.words(), output_vocab.words());					

					best_model = epoch+1;
				}				
			}
			//scale the model back for training
			//scale the model before writing it
			//input.scale(1./(1.-myParam.dropout_probability));
			decoder_input.scale(1./(1-myParam.dropout_probability));
			nn_decoder.scale(1./(1-myParam.dropout_probability));
			if (myParam.max_epoch > -1 && epoch+1 >= myParam.max_epoch) {
				current_learning_rate /= 2;
			} else  
			//if (epoch > 0 && log_likelihood < current_validation_ll && myParam.parameter_update != "ADA")
			if (epoch > 0 && current_accuracy < accuracy && myParam.parameter_update != "ADA")	
		        { 
		            current_learning_rate /= 2;
		        }
	        current_validation_ll = log_likelihood;
			current_accuracy = accuracy;
	 
		}
	
    }
	cerr<<" The best validation perplexity achieved in epoch "<<best_model<<" was "<<best_perplexity<<" and the models are ";
		if (arg_run_lm == 1) {
			cerr<<myParam.model_prefix<<".decoder.best"<<endl;
		} else {
			cerr<<myParam.model_prefix<<".encoder.best"<<", "<<myParam.model_prefix<<".decoder.best"<<endl;
		}
    return 0;
}

