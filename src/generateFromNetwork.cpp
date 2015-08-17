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
#include "define.h"
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
typedef unordered_map<Matrix<int,Dynamic,1>, precision_type> vector_map;

typedef ip::allocator<int, ip::managed_mapped_file::segment_manager> intAllocator;
typedef ip::vector<int, intAllocator> vec;
typedef ip::allocator<vec, ip::managed_mapped_file::segment_manager> vecAllocator;


//typedef long long int data_size_t; // training data can easily exceed 2G instances

int main(int argc, char** argv)
{ 
	srand (time(NULL));
	setprecision(16);
    ios::sync_with_stdio(false);
    bool use_mmap_file, randomize, tagging_mode;
    param myParam;
	int arg_output_start_symbol;
	int arg_output_end_symbol;
	string arg_predicted_sequence_file;
	bool arg_greedy;
	bool arg_stochastic;
	bool arg_score;
	bool arg_run_lm;
    try {
      // program options //
      CmdLine cmd("Trains a LSTM.", ' ' , "0.3\n","");

      // The options are printed in reverse order


      //ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 64.", false, 64, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 1000.", false, 100, "int", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
      //ValueArg<int> num_hidden("", "num_hidden", "Number of hidden nodes. Default: 100. All gates, cells, hidden layers, \n \
		  							input and output embedding dimension are set to this value", false, 100, "int", cmd);

      //ValueArg<int> output_embedding_dimension("", "output_embedding_dimension", "Number of output embedding dimensions. Default: 50.", false, 50, "int", cmd);
      //ValueArg<int> input_embedding_dimension("", "input_embedding_dimension", "Number of input embedding dimensions. Default: 50.", false, 50, "int", cmd);
      //ValueArg<int> embedding_dimension("", "embedding_dimension", "Number of input and output embedding dimensions. Default: none.", false, -1, "int", cmd);

      //ValueArg<int> vocab_size("", "vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
	  //ValueArg<int> output_start_symbol("", "output_start_symbol", "The integer id of the output start symbol. Default: 0.", false, 0, "int", cmd);
	  //ValueArg<int> output_end_symbol("", "output_end_symbol", "The integer id of the output end symbol Default: 1.", false, 1, "int", cmd);
      //ValueArg<int> input_vocab_size("", "input_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      //ValueArg<int> output_vocab_size("", "output_vocab_size", "Vocabulary size. Default: auto.", false, 0, "int", cmd);
      //ValueArg<int> ngram_size("", "ngram_size", "Size of n-grams. Default: auto.", false, 0, "int", cmd);


      //ValueArg<string> input_words_file("", "input_words_file", "Vocabulary." , false, "", "string", cmd);
      //ValueArg<string> output_words_file("", "output_words_file", "Vocabulary." , false, "", "string", cmd);
	  //ValueArg<string> input_sent_file("", "input_sent_file", "Input sentences file." , false, "", "string", cmd);
	  //ValueArg<string> output_sent_file("", "output_sent_file", "Input sentences file." , false, "", "string", cmd);
	  ValueArg<string> testing_sent_file("", "testing_sent_file", "Input sentences file." , true, "", "string", cmd);
	  //ValueArg<string> testing_sequence_cont_file("", "testing_sequence_cont_file", "Testing sequence continuation file" , false, "", "string", cmd);


	  //ValueArg<bool> restart_states("", "restart_states", "If yes, then the hidden and cell values will be restarted after every minibatch \n \
		//  Default: 1 = yes, \n \
		 // 			0 = gradient clipping. Default: 0.", false, 1, "bool", cmd);	  
      ValueArg<string> encoder_model_file("", "encoder_model_file", "Encoder Model file.", false, "", "string", cmd);
	  ValueArg<string> decoder_model_file("", "decoder_model_file", "Decoder Model file.", false, "", "string", cmd);
	  //ValueArg<precision_type> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "precision_type", cmd);
	  ValueArg<string> predicted_sequence_file("", "predicted_sequence_file", "Predicted sequences file." , false, "", "string", cmd);
	  ValueArg<bool> greedy("", "greedy", "If yes, then the output will be generated greedily \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	
	  ValueArg<bool> stochastic("", "stochastic", "If yes, then the output will be generated stochastically \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	
	  ValueArg<bool> score("", "score", "If yes, then the program will compute the log probability of output given input \n \
		  or probability of sentence if run as a language model. Default: 0 = no. \n", false, 0, "bool", cmd);	  	  	  
	  ValueArg<bool> run_lm("", "run_lm", "Run as a language model, \n \
		  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);		  
      cmd.parse(argc, argv);


      myParam.encoder_model_file = encoder_model_file.getValue();
	  myParam.decoder_model_file = decoder_model_file.getValue();
      //myParam.train_file = train_file.getValue();
      //myParam.validation_file = validation_file.getValue();
      //myParam.input_words_file = input_words_file.getValue();
      //myParam.output_words_file = output_words_file.getValue();
	  //myParam.input_sent_file = input_sent_file.getValue();
	  //myParam.output_sent_file = output_sent_file.getValue();
	  myParam.testing_sent_file = testing_sent_file.getValue();
	  //arg_output_start_symbol = output_start_symbol.getValue();
	  //arg_output_end_symbol = output_end_symbol.getValue();
	  arg_predicted_sequence_file = predicted_sequence_file.getValue();
	  arg_greedy = greedy.getValue();
	  arg_stochastic = stochastic.getValue();
	  arg_score = score.getValue();
	  arg_run_lm = run_lm.getValue();
	  
	  /*
	  if (arg_greedy == 0 && arg_stochastic == 0 && arg_score == 0){
		  cerr<<"You have to choose either stocastic or greedy generation or to score"<<endl;
		  exit(0);
	  }
	  if (arg_greedy == 1 && arg_stochastic == 1){
	  	cerr<<"You have to choose either stocastic or greedy generation, not both"<<endl;
		exit(1);
	  }
	  if (arg_greedy == 1){
		  arg_stochastic = 0;
	  }
	  if (arg_stochastic == 1){
		  arg_greedy = 0;
	  }
	  */
	  
	  if (arg_greedy + arg_stochastic + arg_score  == 0 || arg_greedy + arg_stochastic + arg_score  >= 2 ) {
		  cerr<<"You have to choose only one option between greedy, stochastic and score"<<endl;
		  cerr<<"Currently : "<<endl;
		  cerr<<"greedy     :"<<greedy.getValue()<<endl;
		  cerr<<"stochastic :"<<stochastic.getValue()<<endl;
		  cerr<<"score      :"<<score.getValue()<<endl;
		  exit(1); 
	  }
	  //myParam.testing_sequence_cont_file = testing_sequence_cont_file.getValue();
	  //myParam.validation_sequence_cont_file = validation_sequence_cont_file.getValue();
	  /*
      if (words_file.getValue() != "")
	      myParam.input_words_file = myParam.output_words_file = words_file.getValue();
	  */
      //myParam.model_prefix = model_prefix.getValue();

      //myParam.ngram_size = ngram_size.getValue();
      //myParam.vocab_size = vocab_size.getValue();
      //myParam.input_vocab_size = input_vocab_size.getValue();
      //myParam.output_vocab_size = output_vocab_size.getValue();
	  myParam.num_threads = num_threads.getValue();
	  
	  /*
      if (vocab_size.getValue() >= 0) {
	      myParam.input_vocab_size = myParam.output_vocab_size = vocab_size.getValue();
      }
      myParam.num_hidden = num_hidden.getValue();

      myParam.input_embedding_dimension = input_embedding_dimension.getValue();
      myParam.output_embedding_dimension = output_embedding_dimension.getValue();
      if (embedding_dimension.getValue() >= 0) {
	      myParam.input_embedding_dimension = myParam.output_embedding_dimension = embedding_dimension.getValue();
      }
	  */
	  
      myParam.minibatch_size = minibatch_size.getValue();
	  //myParam.minibatch_size = 1; //hard coding this for now
	  
      //myParam.validation_minibatch_size = validation_minibatch_size.getValue();

	  //myParam.restart_states = norm_threshold.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");
	  /*
      //cerr << train_file.getDescription() << sep << train_file.getValue() << endl;
      //cerr << validation_file.getDescription() << sep << validation_file.getValue() << endl;
      cerr << input_words_file.getDescription() << sep << input_words_file.getValue() << endl;
      cerr << output_words_file.getDescription() << sep << output_words_file.getValue() << endl;
      //cerr << model_prefix.getDescription() << sep << model_prefix.getValue() << endl;

      cerr << ngram_size.getDescription() << sep << ngram_size.getValue() << endl;
      cerr << input_vocab_size.getDescription() << sep << input_vocab_size.getValue() << endl;
      cerr << output_vocab_size.getDescription() << sep << output_vocab_size.getValue() << endl;

	  //cerr << restart_states.getDescription() <<sep <<restart_states.getValue() <<endl;

      if (embedding_dimension.getValue() >= 0)
      {
	      cerr << embedding_dimension.getDescription() << sep << embedding_dimension.getValue() << endl;
      }
      else
      {
	      cerr << input_embedding_dimension.getDescription() << sep << input_embedding_dimension.getValue() << endl;
	      cerr << output_embedding_dimension.getDescription() << sep << output_embedding_dimension.getValue() << endl;
      }
	  */
	  
	  /*
      cerr << share_embeddings.getDescription() << sep << share_embeddings.getValue() << endl;
      if (share_embeddings.getValue() && input_embedding_dimension.getValue() != output_embedding_dimension.getValue())
      {
	      cerr << "error: sharing input and output embeddings requires that input and output embeddings have same dimension" << endl;
	      exit(1);
      }
	  */
      //cerr << num_hidden.getDescription() << sep << num_hidden.getValue() << endl;
	  //cerr<<input_sent_file.getDescription() << sep << input_sent_file.getValue() << endl;
	  //cerr<<output_sent_file.getDescription() << sep << output_sent_file.getDescription() <<endl;
	  cerr<<testing_sent_file.getDescription() << sep << testing_sent_file.getValue() <<endl;
	  cerr<<encoder_model_file.getDescription() << sep << encoder_model_file.getValue() <<endl;
	  cerr<<decoder_model_file.getDescription() << sep << decoder_model_file.getValue() <<endl;
	  cerr << predicted_sequence_file.getDescription()<< sep << predicted_sequence_file.getValue() << endl;
	  
      cerr << minibatch_size.getDescription() << sep << minibatch_size.getValue() << endl;
	  cerr << greedy.getDescription()<< sep << greedy.getValue() << endl;
	  cerr << stochastic.getDescription()<< sep << stochastic.getValue() << endl;
	  cerr << score.getDescription()<< sep << score.getValue() << endl;
	  cerr << run_lm.getDescription()<< sep << run_lm.getValue() << endl;
	  
      //if (myParam.validation_file != "") {
	  //   cerr << validation_minibatch_size.getDescription() << sep << validation_minibatch_size.getValue() << endl;
     // }

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
    mt19937 rng(time(NULL));

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////

    // Read training data

    vector<int> training_data_flat;
	vector< vector<int> > testing_input_sent, testing_sequence_cont_sent;
	vector< vector<int> > testing_output_sent , validation_sequence_cont_sent;
	
	vector< vector<string> > word_testing_input_sent, word_testing_output_sent;
    vec * training_data_flat_mmap;
    data_size_t testing_data_size, validation_data_size; //num_tokens;
    ip::managed_mapped_file mmap_file;
	/*
    if (use_mmap_file == false) {
      cerr<<"Reading data from regular text file "<<endl;
      readDataFile(myParam.train_file, myParam.ngram_size, training_data_flat, myParam.minibatch_size);
      training_data_size = training_data_flat.size()/myParam.ngram_size;
    } else {
 
    }
	
	*/
	if (arg_run_lm == 1) {
		cerr<<"Running as a LSTM language model"<<endl;
	} else {
		cerr<<"Running as a LSTM sequence to sequence model"<<endl;
	}	
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
	//readSentFile(myParam.input_sent_file, testing_input_sent,myParam.minibatch_size, total_input_tokens);
	if (arg_run_lm == 0) { //If you're running in LM mode, you do not need to read in the input
		readEvenSentFile(myParam.testing_sent_file, word_testing_input_sent, total_input_tokens,1,0);
	}
	if (arg_score) {
		readOddSentFile(myParam.testing_sent_file, word_testing_output_sent, total_output_tokens,1,1);	
	}
	//readSentFile(myParam.output_sent_file, testing_output_sent,myParam.minibatch_size, total_output_tokens);
    //readSentFile(myParam.testing_sequence_cont_file, testing_sequence_cont_sent, myParam.minibatch_size, total_training_sequence_tokens);
	//exit(0);
	if (arg_run_lm == 1)
		testing_data_size = word_testing_output_sent.size();
	else 
		testing_data_size = word_testing_input_sent.size();

	data_size_t num_batches = (testing_data_size-1)/myParam.minibatch_size + 1;

    cerr<<"Number of input tokens "<<total_input_tokens<<endl;
	cerr<<"Number of output tokens "<<total_output_tokens<<endl;
	//cerr<<"Number of minibatches "<<num_batches<<endl;
    //data_size_t training_data_size = num_tokens / myParam.ngram_size;
	
    cerr << "Number of testing instances "<< word_testing_input_sent.size() << endl;
    //cerr << "Number of validation instances "<< validation_input_sent.size() << endl;
	
    //Matrix<int,Dynamic,Dynamic> testing_data;
	Matrix<int,Dynamic,Dynamic> testing_input_sent_data, testing_output_sent_data;
	//Matrix<int,Dynamic,Dynamic> validation_input_sent_data, validation_output_sent_data;
	//Array<int,Dynamic,Dynamic> testing_sequence_cont_sent_data;
	Array<int,Dynamic,Dynamic> testing_input_sequence_cont_sent_data;
	Array<int,Dynamic,Dynamic> testing_output_sequence_cont_sent_data;	
    //(training_data_flat.data(), myParam.ngram_size, training_data_size);
    
    ///// Read in vocabulary file. We don't actually use it; it just gets reproduced in the output file

	/*
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
	*/
	


    ///// Create and initialize the neural network and associated propagators.
	vector<string> encoder_input_words, encoder_output_words;
	vector<string> decoder_input_words, decoder_output_words;
	
    model encoder_nn,decoder_nn;
	encoder_nn.read(myParam.encoder_model_file, encoder_input_words, encoder_output_words);
	decoder_nn.read(myParam.decoder_model_file, decoder_input_words, decoder_output_words);
	
	vocabulary encoder_vocab(encoder_input_words), decoder_vocab(decoder_output_words);

	arg_output_start_symbol = decoder_vocab.lookup_word("<s>");
	arg_output_end_symbol = decoder_vocab.lookup_word("</s>");
	
	//cerr<<"The symbol <s> has id "<<arg_output_start_symbol<<endl;
	//cerr<<"The symbol </s> has id "<<arg_output_end_symbol<<endl;
	google_input_model encoder_input, decoder_input;

	encoder_input.read(myParam.encoder_model_file);
	decoder_input.read(myParam.decoder_model_file);
	encoder_nn.set_input(encoder_input);
	decoder_nn.set_input(decoder_input);
	
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL
	myParam.num_hidden = encoder_nn.get_hidden();
	myParam.input_embedding_dimension = myParam.num_hidden;
	myParam.output_embedding_dimension = myParam.num_hidden;
	cerr<<"done reading the models "<<endl;
	
	//Transforming the input and output data
	if (arg_run_lm == 0) {
		integerize(word_testing_input_sent, 
						testing_input_sent, 
						encoder_vocab);
	}
	if (arg_score) { 
		integerize(word_testing_output_sent, 
						testing_output_sent, 
						decoder_vocab);	
	}
	//cerr<<"Num hidden is "<<myParam.num_hidden<<endl;
	//cerr<<"minibatch size is "<<myParam.minibatch_size<<endl;
    //loss_function_type loss_function = string_to_loss_function(myParam.loss_function);

    //propagator prop(nn, myParam.minibatch_size);
	propagator<Google_input_node, google_input_model> prop(encoder_nn, decoder_nn, myParam.minibatch_size);

    ///////////////////////TESTING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

	//string temp_encoder_file = "temp.encoder";
	//string temp_decoder_file = "temp.decoder";
    cerr<<"Number of testing minibatches: "<<num_batches<<endl;
	
	/*
    int num_validation_batches = 0;
    if (validation_data_size > 0)
    {
        num_validation_batches = (validation_data_size-1)/myParam.validation_minibatch_size+1;
		cerr<<"Number of validation minibatches: "<<num_validation_batches<<endl;
    } 
	*/
	


    int ngram_size = myParam.ngram_size;
    int input_vocab_size = myParam.input_vocab_size;
    int output_vocab_size = myParam.output_vocab_size;
    int minibatch_size = myParam.minibatch_size;
    //int validation_minibatch_size = myParam.validation_minibatch_size;
    //int num_noise_samples = myParam.num_noise_samples;

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
    //for (int epoch=0; epoch<myParam.num_epochs; epoch++)
    //{ 
    //    cerr << "Epoch " << epoch+1 << endl;
   //     cerr << "Current learning rate: " << current_learning_rate << endl;

   //      if (myParam.use_momentum) 
   //	    cerr << "Current momentum: " << current_momentum << endl;
	//else
    //        current_momentum = -1;

	cerr << "Testing minibatches: ";

	precision_type log_likelihood = 0.0;
	

	
	//loss_function = LogLoss;

	
	//cerr<<"Training data size is"<<training_data_size<<endl;
    //data_size_t num_batches = (training_data_size-1)/myParam.minibatch_size + 1;
	precision_type data_log_likelihood=0;	
	Matrix<precision_type,Dynamic,Dynamic> current_c_for_gradCheck, current_h_for_gradCheck, current_c,current_h, init_c, init_h;
	current_c.setZero(myParam.num_hidden, minibatch_size);
	current_h.setZero(myParam.num_hidden, minibatch_size);
	//init_c.setZero(myParam.num_hidden,minibatch_size);
	//init_h.setZero(myParam.num_hidden,minibatch_size);
	//c_last.setZero(numParam.num_hidden, minibatch_size);
	//h_last.setZero(numParam.num_hidden, minibatch_size);
	//encoder_nn.write(temp_encoder_file);
	//decoder_nn.write(temp_decoder_file);
	
	ofstream file;
	if (arg_predicted_sequence_file != "") {
		//cerr<<"opeining a file"<<endl;
    	file.open(arg_predicted_sequence_file.c_str(), std::ofstream::out ); //we are not appending| std::ofstream::app);
		//file << std::setprecision(15);
    	if (!file) throw runtime_error("Could not open file " + arg_predicted_sequence_file);
		//file <<"BLAH!"<<endl;
	}

	//precision_type log_likelihood = 0.0;
	
    for(data_size_t batch=0;batch<num_batches;batch++)
    {
		
			current_c.setZero(myParam.num_hidden, minibatch_size);
			current_h.setZero(myParam.num_hidden, minibatch_size);		
			precision_type minibatch_log_likelihood = 0.;
            if (batch > 0 && batch % 100 == 0)
            {
	        cerr << batch <<"...";
            } 

            data_size_t minibatch_start_index = minibatch_size * batch;
			data_size_t minibatch_end_index = min(testing_data_size-1, static_cast<data_size_t> (minibatch_start_index+minibatch_size-1));
			//cerr<<"Minibatch start index is "<<minibatch_start_index<<endl;
			//cerr<<"Minibatch end index is "<<minibatch_end_index<<endl;

      	  int current_minibatch_size = min(static_cast<data_size_t>(minibatch_size), testing_data_size - minibatch_start_index);
		  vector<vector<int> > predicted_sequence(current_minibatch_size);
	  	  //cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;

	  

		  	//precision_type adjusted_learning_rate = current_learning_rate;
			//cerr<<"Adjusted learning rate is"<<adjusted_learning_rate<<endl;
            //cerr<<"Adjusted learning rate: "<<adjusted_learning_rate<<endl;

            ///// Forward propagation

			//Taking the input and output sentence and setting the testing data to it.
			//Getting a minibatch of sentences
			vector<int> minibatch_input_sentences, 
				minibatch_output_sentences, 
				minibatch_input_sequence_cont_sentences,
				minibatch_output_sequence_cont_sentences;
			unsigned int max_input_sent_len, max_output_sent_len;
			max_input_sent_len = max_output_sent_len = 0;
			unsigned int minibatch_output_tokens,minibatch_input_tokens, minibatch_sequence_cont_tokens;
			minibatch_output_tokens = minibatch_input_tokens = minibatch_sequence_cont_tokens = 0;
			/*
			miniBatchifyEncoder(testing_input_sent, 
							minibatch_input_sentences,
							minibatch_start_index,
							minibatch_end_index,
							max_input_sent_len,
							1,
							minibatch_input_tokens);
			*/
			if (arg_run_lm == 0) {
				miniBatchifyEncoder(testing_input_sent, 
								minibatch_input_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_input_sent_len,
								minibatch_input_tokens,
								1);	
				minibatch_input_tokens = 0;
				miniBatchifyEncoder(testing_input_sent, 
								minibatch_input_sequence_cont_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_input_sent_len,
								minibatch_input_tokens,
								0);		
						
				testing_input_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_input_sentences.data(), 
												max_input_sent_len,
												current_minibatch_size);
				testing_input_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_input_sequence_cont_sentences.data(),
																				max_input_sent_len,
																				current_minibatch_size);
			}																
			if (arg_score == 1) {
				miniBatchifyDecoder(testing_output_sent, 
								minibatch_output_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1);
			
				minibatch_output_tokens =0;
				miniBatchifyDecoder(testing_output_sent, 
								minibatch_output_sequence_cont_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								0);		
				testing_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_output_sentences.data(),
																				max_output_sent_len,
																				current_minibatch_size);

				testing_output_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_output_sequence_cont_sentences.data(),
																			max_output_sent_len,
																			current_minibatch_size);											
			}																			
			//cerr<<"sequence cont data is "<<testing_sequence_cont_sent_data<<endl;
			//testing_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_output_sentences.data(),
			//																max_output_sent_len,
			//																current_minibatch_size);
			//testing_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_sequence_cont_sentences.data(),
			//																max_sent_len,
			//																current_minibatch_size);
																											
			//testing_sequence_cont_sent_data = Array<int,Dynamic,Dynamic>();																																			
			init_c = current_c;
			init_h = current_h; 	
			if (arg_run_lm == 0) {		
				prop.fPropEncoder(testing_input_sent_data,
							0,
							max_input_sent_len-1,
							current_c,
							current_h,
							testing_input_sequence_cont_sent_data);	
			}
			//prop.computeProbsLog(testing_output_sent_data,
			// 					minibatch_log_likelihood);	
			//cerr<<"output start symbol "<<arg_output_start_symbol<<endl;
			//cerr<<"output end symbol "<<arg_output_end_symbol<<endl;
			if (arg_greedy) {
				//cerr<<"Generating greedy output"<<endl;
				prop.generateGreedyOutput(testing_input_sent_data,
								current_c,
								current_h,		
								predicted_sequence,
								arg_output_start_symbol,
								arg_output_end_symbol);				
			}
			if (arg_stochastic){
				prop.generateStochasticOutput(testing_input_sent_data,
								current_c,
								current_h,		
								predicted_sequence,
								arg_output_start_symbol,
								arg_output_end_symbol,
								rng);
			}
			if (arg_score) {
			    prop.fPropDecoder(testing_output_sent_data,
						current_c,
						current_h,
						testing_output_sequence_cont_sent_data);	
											 
		 		prop.computeProbsLog(testing_output_sent_data,
		 		 			  	minibatch_log_likelihood);
				//cerr<<"Minibatch log likelihood is "<<minibatch_log_likelihood<<endl;
				log_likelihood += minibatch_log_likelihood;				
			}
			//cerr<<"predicted sequence size is "<<predicted_sequence.size()<<endl;
			//data_log_likelihood += 	minibatch_log_likelihood;
			//writing the predicted sequence
			if (arg_stochastic || arg_greedy) {
				//cerr<<"writing to file "<<endl;
				//cerr<<"predicted sequence length is "<<predicted_sequence.size()<<endl;
				for (int sent_id = 0; sent_id<predicted_sequence.size(); sent_id++) {
					for (int seq_id=0; seq_id<predicted_sequence[sent_id].size(); seq_id++){
						//cerr<<"sequence word is "<<decoder_vocab.get_word(predicted_sequence[sent_id][seq_id])<<endl;
						file << decoder_vocab.get_word(predicted_sequence[sent_id][seq_id])<<" ";	
					}
					file<<endl;
				}
			}

		  
	 }
	 if (arg_predicted_sequence_file != ""){
		 file.close();
	 }
	 cerr << "done." << endl;
	 if(arg_score) {
        //cerr << "Validation log-likelihood: "<< log_likelihood << endl;
        //cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;
	    cerr << "		Testing log-likelihood base e    :   " << log_likelihood << endl;
		cerr << "		Testing log-likelihood base 2    :   " << log_likelihood/log(2.) << endl;
		cerr<<  "		Testing cross entropy in base 2  :   "<< log_likelihood/(log(2.)*total_output_tokens)<< endl;
		cerr << "         		perplexity               :   "<< exp(-log_likelihood/total_output_tokens) << endl;	 	
	 } 	 

	//if (loss_function == LogLoss)
	//{
		//cerr<<"log likelihood base e is"<<log_likelihood<<endl;
		//cerr<<"log likelihood base 10 is"<<log_likelihood/log(10.)<<endl;
		//cerr<<"The cross entopy in base 10 is "<<log_likelihood/(log(10.)*sent_len)<<endl;
		//cerr<<"The training perplexity is "<<exp(-log_likelihood/sent_len)<<endl;
		//log_likelihood /= sent_len;		
	    //cerr << "Testing log-likelihood base e:      " << data_log_likelihood << endl;
		//cerr << "Testing log-likelihood base 2:     " << data_log_likelihood/log(2.) << endl;
		//cerr << "Testing cross entropy in base 2 is "<<data_log_likelihood/(log(2.)*total_output_tokens)<< endl;
		//cerr << "         perplexity:                 "<< exp(-data_log_likelihood/total_output_tokens) << endl;
	//}
	//else if (loss_function == NCELoss)
	//   cerr << "Testing NCE log-likelihood: " << log_likelihood << endl;
	
	//if (myParam.use_momentum)
    //    current_momentum += momentum_delta;

	#ifdef USE_CHRONO
	cerr << "Propagation times:";
	for (int i=0; i<timer.size(); i++)
	  cerr << " " << timer.get(i);
	cerr << endl;
	#endif
	
	

	
    //}
    return 0;
}

