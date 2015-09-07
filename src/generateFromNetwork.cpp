//TODO: 
//1. beam search should allow minibatches. It should be possible
//2. Generate hiddens states is very ugly right now. It assumes so much about the architecture.
// It should accept a struct back and then print the struct

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


template<typename VectorDerivedO, 
			typename VectorDerivedF, 
			typename VectorDerivedI, 
			typename VectorDerivedH, 
			typename VectorDerivedC>
void writeStates(ofstream &hidden_states_file,
					const vector<VectorDerivedH >   &internal_h_t,
					const vector<VectorDerivedC>   &internal_c_t,
					const vector<VectorDerivedF>   &internal_f_t,
					const vector<VectorDerivedI>   &internal_i_t,
					const vector<VectorDerivedO>   &internal_o_t,
					const int minibatch_index,
					const int word_index){
	hidden_states_file<<"h_t: ";
	writeMatrix(internal_h_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"c_t: ";
	writeMatrix(internal_c_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"o_t: ";
	writeMatrix(internal_o_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"f_t: ";
	writeMatrix(internal_f_t[minibatch_index].col(word_index).transpose(),hidden_states_file);
	hidden_states_file<<"i_t: ";
	writeMatrix(internal_i_t[minibatch_index].col(word_index).transpose(),hidden_states_file);			
}
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
	string arg_predicted_sequence_file, arg_hidden_states_file;
	bool arg_greedy;
	bool arg_stochastic;
	bool arg_score;
	bool arg_run_lm;
	bool generate_hidden_states = 0;
	bool do_beam_search;
	int arg_beam_size = 0;
	bool arg_reverse_input;
	
    try {
      // program options //
      CmdLine cmd("Trains a LSTM.", ' ' , "0.3\n","");

      // The options are printed in reverse order


      //ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "Minibatch size for validation. Default: 64.", false, 64, "int", cmd);
      ValueArg<int> minibatch_size("", "minibatch_size", "Minibatch size (for training). Default: 1000.", false, 100, "int", cmd);

      ValueArg<int> num_threads("", "num_threads", "Number of threads. Default: maximum.", false, 0, "int", cmd);
 
	  ValueArg<string> testing_sent_file("", "testing_sent_file", "Input sentences file." , true, "", "string", cmd);
	  //ValueArg<string> testing_sequence_cont_file("", "testing_sequence_cont_file", "Testing sequence continuation file" , false, "", "string", cmd);


	  //ValueArg<bool> restart_states("", "restart_states", "If yes, then the hidden and cell values will be restarted after every minibatch \n \
		//  Default: 1 = yes, \n \
		 // 			0 = gradient clipping. Default: 0.", false, 1, "bool", cmd);	  
      ValueArg<string> encoder_model_file("", "encoder_model_file", "Encoder Model file. Default: empty", false, "", "string", cmd);
	  ValueArg<string> decoder_model_file("", "decoder_model_file", "Decoder Model file. Default: empty", false, "", "string", cmd);
	  //ValueArg<precision_type> norm_threshold("", "norm_threshold", "Threshold for gradient norm. Default 5", false,5., "precision_type", cmd);
	  ValueArg<string> predicted_sequence_file("", "predicted_sequence_file", "Predicted sequences file." , false, "", "string", cmd);
	  ValueArg<bool> greedy("", "greedy", "If yes, then the output will be generated greedily \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	
	  ValueArg<bool> stochastic("", "stochastic", "If yes, then the output will be generated stochastically \n \
		  Default: 0 = no. \n", false, 0, "bool", cmd);	
	  ValueArg<int> beam_size("", "beam_size", "Do beam search with specified beam size (>=1). Default: 0 (No beam search). \n \
	Note: Please supply an argument to --predicted_sequence_file where the results of beam search will be saved.", false, 0, "int", cmd);		  
	  ValueArg<bool> score("", "score", "If yes, then the program will compute the log probability of output given input \n \
		  or probability of sentence if run as a language model. Default: 0 = no. \n", false, 0, "bool", cmd);	  	  	  
	  ValueArg<bool> run_lm("", "run_lm", "Run as a language model, \n \
		  			1 = yes. Default: 0 (Run as a sequence to sequence model).", false, 0, "bool", cmd);	
	  ValueArg<bool> reverse_input("", "reverse", "Reverse the input sentence before decoding. \n \
		  			It should be set to the same value as used during training. \n \
		  			1 = yes. Default: 0 (No reversing).", false, 0, "bool", cmd);  
	  ValueArg<string> hidden_states_file("", "hidden_states_file", "Dump the hidden states in this file. Will currently only \
		  generate the forced decode trace. That is, it needs both input and output pairs. Default: empty", false, "", "string", cmd);	  
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
	  arg_hidden_states_file = hidden_states_file.getValue();
	  arg_beam_size = beam_size.getValue();
	  do_beam_search = (arg_beam_size > 0);
	  arg_reverse_input = reverse_input.getValue();

	  
	  if (arg_hidden_states_file != "")
		  generate_hidden_states = 1;
	  if (arg_greedy + arg_stochastic + arg_score + generate_hidden_states + do_beam_search == 0 || arg_greedy + arg_stochastic + arg_score + do_beam_search >= 2 ) {
		  cerr<<"You have to choose only one option between greedy, stochastic, score, generating hidden states, or do beam search"<<endl;
		  cerr<<"Currently             : "<<endl;
		  cerr<<"greedy                :"<<greedy.getValue()<<endl;
		  cerr<<"stochastic            :"<<stochastic.getValue()<<endl;
		  cerr<<"score                 :"<<score.getValue()<<endl;
		  cerr<<"hidden states file    :"<<hidden_states_file.getValue()<<endl;
		  cerr<<"beam size             :"<<beam_size.getValue()<<endl;
		  exit(1); 
	  }
	  if (arg_greedy + arg_stochastic + do_beam_search >= 1 && arg_predicted_sequence_file == ""){
		  cerr<<"You have to specify a predicted sequence file!"<<endl;
		  exit(1);
	  }


	  myParam.num_threads = num_threads.getValue();
	  

	  
      myParam.minibatch_size = minibatch_size.getValue();
	  if (do_beam_search) {
		  myParam.minibatch_size = 1;
		  //myParam.minibatch_size = arg_beam_size;
	  }
	  		
	  //myParam.minibatch_size = 1; //hard coding this for now
	  
      //myParam.validation_minibatch_size = validation_minibatch_size.getValue();

	  //myParam.restart_states = norm_threshold.getValue();

      cerr << "Command line: " << endl;
      cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;

      const string sep(" Value: ");

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
	  cerr << reverse_input.getDescription()<< sep << reverse_input.getValue() << endl;
	  cerr << hidden_states_file.getDescription() << sep << hidden_states_file.getValue() << endl;
	  cerr << beam_size.getDescription() << sep << beam_size.getValue() <<endl;
	  
      //if (myParam.validation_file != "") {
	  //   cerr << validation_minibatch_size.getDescription() << sep << validation_minibatch_size.getValue() << endl;
     // }

    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }
	vector<precision_type> sentence_probabilities; 
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
	vector< vector<int> > decoder_testing_output_sent, decoder_testing_input_sent;
	
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
		readEvenSentFile(myParam.testing_sent_file, word_testing_input_sent, total_input_tokens,1,0, arg_reverse_input);
	}
	if (arg_score || generate_hidden_states) {
		readOddSentFile(myParam.testing_sent_file, word_testing_output_sent, total_output_tokens,1,1,0);	
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
	Matrix<int,Dynamic,Dynamic> decoder_testing_output_sent_data, decoder_testing_input_sent_data;
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
	google_input_model encoder_input, decoder_input;
	vocabulary encoder_vocab, decoder_vocab, decoder_input_vocab, decoder_output_vocab;
	//if (arg_run_lm == 0) {
		encoder_nn.read(myParam.encoder_model_file, encoder_input_words, encoder_output_words);
		encoder_input.read(myParam.encoder_model_file);
		encoder_nn.set_input(encoder_input);
		encoder_vocab.build_vocab(encoder_input_words);
	//}
	
	decoder_nn.read(myParam.decoder_model_file, decoder_input_words, decoder_output_words);
	//vocabulary encoder_vocab(encoder_input_words), decoder_vocab(decoder_output_words);
	decoder_input_vocab.build_vocab(decoder_input_words);
	decoder_output_vocab.build_vocab(decoder_output_words);
	
	arg_output_start_symbol = decoder_input_vocab.lookup_word("<s>");
	arg_output_end_symbol = decoder_output_vocab.lookup_word("</s>");
	
	//cerr<<"The symbol <s> has id "<<arg_output_start_symbol<<endl;
	//cerr<<"The symbol </s> has id "<<arg_output_end_symbol<<endl;
	
	if (arg_beam_size > decoder_vocab.size()) {
		cerr<<"Warning: The beam size is larger than the output vocabulary."<<endl;
		cerr<<"User specified beam size: "<<arg_beam_size<<endl;
		cerr<<"Output vocabulary size: "<<decoder_vocab.size()<<endl;
		//arg_beam_size  = decoder_vocab.size();
		//cerr<<"New beam size :"<<arg_beam_size<<endl;
	}

	decoder_input.read(myParam.decoder_model_file);
	
	decoder_nn.set_input(decoder_input);
	
    // IF THE MODEL FILE HAS BEEN DEFINED, THEN 
    // LOAD THE NEURAL NETWORK MODEL
	myParam.num_hidden = decoder_nn.get_hidden();
	myParam.input_embedding_dimension = myParam.num_hidden;
	myParam.output_embedding_dimension = myParam.num_hidden;
	cerr<<"done reading the models "<<endl;
	
	//Transforming the input and output data
	if (arg_run_lm == 0) {
		integerize(word_testing_input_sent, 
						testing_input_sent, 
						encoder_vocab);
	}
	if (arg_score || generate_hidden_states) { 
		integerize(word_testing_output_sent, 
						testing_output_sent, 
						decoder_vocab);	
		integerize(word_testing_output_sent, 
				   decoder_testing_output_sent,
				   decoder_output_vocab,
				   1,
				   0);
	   	integerize(word_testing_output_sent, 
	   			   decoder_testing_input_sent,
	   			   decoder_input_vocab,
	   			   0,
	   			   1);								
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
    //cerr<<"Number of testing minibatches: "<<num_batches<<endl;
	
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
	ofstream hidden_states_file;
	if (arg_hidden_states_file != ""){
		//if (arg_score == 0) {
		//	cerr<<"Error! You can dump hidden states with --score 1 "<<endl;
		//	exit(1);
		//}
		hidden_states_file.open(arg_hidden_states_file.c_str(),std::ofstream::out )	;
	} 
	
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
				minibatch_decoder_output_sentences,
				minibatch_decoder_input_sentences,
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
			if (arg_score == 1 || generate_hidden_states == 1) {
				miniBatchifyDecoder(decoder_testing_output_sent, 
								minibatch_decoder_output_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1,
								-1);		
				minibatch_output_tokens =0;						
				miniBatchifyDecoder(decoder_testing_input_sent, 
								minibatch_decoder_input_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1,
								0);									
				decoder_testing_output_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_output_sentences.data(),
																							max_output_sent_len,
																							current_minibatch_size);
				decoder_testing_input_sent_data = Map< Matrix<int, Dynamic, Dynamic> > (minibatch_decoder_input_sentences.data(),
																							max_output_sent_len,
																							current_minibatch_size);	
				/*	
				minibatch_output_tokens =0;																				
				miniBatchifyDecoder(testing_output_sent, 
								minibatch_output_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								1);
			
				

				testing_output_sent_data = Map< Matrix<int,Dynamic,Dynamic> >(minibatch_output_sentences.data(),
																				max_output_sent_len,
																				current_minibatch_size);
				*/
				minibatch_output_tokens =0;
				miniBatchifyDecoder(decoder_testing_input_sent, 
								minibatch_output_sequence_cont_sentences,
								minibatch_start_index,
								minibatch_end_index,
								max_output_sent_len,
								minibatch_output_tokens,
								0);		
				testing_output_sequence_cont_sent_data = Map< Array<int,Dynamic,Dynamic> >(minibatch_output_sequence_cont_sentences.data(),
																			max_output_sent_len,
																			current_minibatch_size);
																														
			}																			
																																			
			init_c = current_c;
			init_h = current_h; 	
			vector<Matrix<precision_type, Dynamic, Dynamic> > input_hiddens,
															encoder_h_t,
															encoder_o_t,
															encoder_c_t,
															encoder_i_t,
															encoder_f_t,
															decoder_h_t,
															decoder_o_t,
															decoder_c_t,
															decoder_i_t,
															decoder_f_t;
															
			vector<Matrix<precision_type, Dynamic, Dynamic> > output_hiddens;
			if (arg_run_lm == 0) {		
				//cerr<<"Current minibatch size is "<<current_minibatch_size<<endl;
				//cerr<<"testing_input_sent_data is "<<testing_input_sent_data<<endl;
				prop.fPropEncoder(testing_input_sent_data,
							0,
							max_input_sent_len-1,
							current_c,
							current_h,
							testing_input_sequence_cont_sent_data);	
				
				//printing out the encoder states along with the sentence
							
				if (arg_hidden_states_file != ""){
					//cerr<<"gettig hiddens from encoder"<<endl;

					for(int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++) {
						Matrix< precision_type, Dynamic, Dynamic> hidden_states,
																	internal_h_t,
																	internal_c_t,
																	internal_i_t,
																	internal_f_t,
																	internal_o_t; 
						hidden_states.setZero(myParam.num_hidden,max_input_sent_len);
						//hidden_states.setZero();
						internal_h_t.setZero(myParam.num_hidden,max_input_sent_len);
						internal_c_t.setZero(myParam.num_hidden,max_input_sent_len);
						internal_i_t.setZero(myParam.num_hidden,max_input_sent_len);
						internal_f_t.setZero(myParam.num_hidden,max_input_sent_len);
						internal_o_t.setZero(myParam.num_hidden,max_input_sent_len);
							
						
						//Getting all the hidden states for the particular sentence
						//cerr<<"max input seq length is "<<max_input_sent_len<<endl;
						prop.getHiddenStates(hidden_states,
										max_input_sent_len,
										1,
										minibatch_index); 
						prop.getInternals(internal_h_t,
										  internal_c_t,
											internal_f_t,
											internal_i_t,
											internal_o_t,
											max_input_sent_len,
											1,
											minibatch_index);										
						//cerr<<"internal h_t"<<internal_h_t<<endl;
						//prob.getInternals	
						input_hiddens.push_back(hidden_states);
						encoder_h_t.push_back(internal_h_t);
						encoder_o_t.push_back(internal_o_t);
						encoder_f_t.push_back(internal_f_t);
						encoder_i_t.push_back(internal_i_t);
						encoder_c_t.push_back(internal_c_t);
					}													
				}
				
				
			}
			//prop.computeProbsLog(testing_output_sent_data,
			// 					minibatch_log_likelihood);	
			//cerr<<"output start symbol "<<arg_output_start_symbol<<endl;
			//cerr<<"output end symbol "<<arg_output_end_symbol<<endl;
			if (arg_greedy) {
				//cerr<<"Generating greedy output"<<endl;
				prop.generateGreedyOutput(testing_input_sent_data,
								decoder_input_vocab,
								decoder_output_vocab,
								current_c,
								current_h,		
								predicted_sequence,
								arg_output_start_symbol,
								arg_output_end_symbol);				
			}
			if (arg_stochastic){
				prop.generateStochasticOutput(testing_input_sent_data,
								decoder_input_vocab,
								decoder_output_vocab,				
								current_c,
								current_h,		
								predicted_sequence,
								arg_output_start_symbol,
								arg_output_end_symbol,
								rng);
			}
			if (do_beam_search) {
				//Need to resize the propagator to the beam size
				prop.resize(arg_beam_size);
				vector<k_best_seq_item> final_k_best_seq_list;
				prop.beamDecoding(testing_input_sent_data,
					decoder_input_vocab,
					decoder_output_vocab,
					current_c,
					current_h,
					final_k_best_seq_list,
					arg_output_start_symbol,
					arg_output_end_symbol,
					arg_beam_size);
				//cerr<<"Final k best seq list size is "<<final_k_best_seq_list.size()<<endl;
				//Now printing the k best list
				file<<"Source\t:";
				//ASSUMING THAT THE MINIBATCH SIZE IS 1 FOR NOW!! HAVE TO CHANGE THIS!!
				//Starting with word index 1 because we don't want to print <s>
				for (int word_index=1; word_index<max_input_sent_len; word_index++){																
					file << encoder_vocab.get_word(testing_input_sent_data(word_index,0))<<" ";
				}
				file<<endl;
				//PRINT if the FINAL K BEST LIST SIZE WAS LESS THAN K!!
				//if (final_k_best_seq_list.size() < arg_beam_size) {

				//}

				//for (int sent_id=0; sent_id<arg_beam_size && ; sent_id++){
				for (int sent_id=0; sent_id<final_k_best_seq_list.size() && sent_id < arg_beam_size ; sent_id++){	
					if (final_k_best_seq_list.size() < arg_beam_size) {
						cerr<<"The n-best list for sentence "<<sent_id<<" has "<<final_k_best_seq_list.size()<<" items"<<endl;
					}
					file<<"Hyp"<<sent_id<<"\t\t:";
					for(int word_id=0; word_id<final_k_best_seq_list.at(sent_id).seq.size(); word_id++){
						if (final_k_best_seq_list.at(sent_id).seq.at(word_id) == arg_output_end_symbol) {
							break;
						} else {
							file << decoder_output_vocab.get_word(final_k_best_seq_list.at(sent_id).seq.at(word_id))<<" ";
						}	
					}
					file<<"Sequence probability: "<<exp(final_k_best_seq_list.at(sent_id).value)<<endl;
					//file<<endl;
				}
				file<<endl;
			}
			if (arg_score || generate_hidden_states ) {
				/*
			    prop.fPropDecoder(testing_output_sent_data,
						current_c,
						current_h,
						testing_output_sequence_cont_sent_data);
				*/
			    prop.fPropDecoder(decoder_testing_input_sent_data,
						current_c,
						current_h,
						testing_output_sequence_cont_sent_data);						
					if (generate_hidden_states) {
						//cerr<<"Getting hiddens from decoder"<<endl;
						for(int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++) {

							//Matrix< precision_type, Dynamic, Dynamic> hidden_states;
							Matrix< precision_type, Dynamic, Dynamic> hidden_states,
																		internal_h_t,
																		internal_c_t,
																		internal_i_t,
																		internal_f_t,
																		internal_o_t; 
							//hidden_states.setZero(myParam.num_hidden,max_input_sent_len);
							//hidden_states.setZero();
							internal_h_t.setZero(myParam.num_hidden,max_input_sent_len);
							internal_c_t.setZero(myParam.num_hidden,max_input_sent_len);
							internal_i_t.setZero(myParam.num_hidden,max_input_sent_len);
							internal_f_t.setZero(myParam.num_hidden,max_input_sent_len);
							internal_o_t.setZero(myParam.num_hidden,max_input_sent_len);						 
							hidden_states.resize(myParam.num_hidden,max_output_sent_len-1);
							hidden_states.setZero();						
						
							prop.getHiddenStates(hidden_states,
											max_output_sent_len-1,
											0,
											minibatch_index); 
										
							prop.getInternals(internal_h_t,
											  internal_c_t,
												internal_f_t,
												internal_i_t,
												internal_o_t,
												max_output_sent_len-1,
												0,
												minibatch_index);	
											
							output_hiddens.push_back(hidden_states);
							//cerr<<"The hidden states are "<<hidden_states<<endl;
							//cerr<<" internal_h_t "<<internal_h_t<<endl;
							decoder_h_t.push_back(internal_h_t);
							decoder_o_t.push_back(internal_o_t);
							decoder_f_t.push_back(internal_f_t);
							decoder_i_t.push_back(internal_i_t);
							decoder_c_t.push_back(internal_c_t);						
						}
					}							
			}
			if (arg_score == 1) {				
				//sentence_probabilities.clear();
				sentence_probabilities = vector<precision_type> (current_minibatch_size,0.);
		 		//prop.computeProbsLog(testing_output_sent_data,
		 		// 			  	minibatch_log_likelihood);
				/*
		 		prop.computeProbsLog(testing_output_sent_data,
		 		 			  	minibatch_log_likelihood,
								sentence_probabilities);	
				*/
		 		prop.computeProbsLog(decoder_testing_output_sent_data,
		 		 			  	minibatch_log_likelihood,
								sentence_probabilities);									
				cerr<<endl;
				for (int sent_id=0; sent_id<current_minibatch_size; sent_id++){
					cerr<<"Sequence "<<minibatch_start_index+sent_id<<" probability: "<<exp(sentence_probabilities[sent_id])<<endl;
				}			
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
					for (int seq_id=0; seq_id<predicted_sequence[sent_id].size()-1 /*we dont want to print </s>*/; seq_id++){
						//cerr<<"sequence word is "<<decoder_vocab.get_word(predicted_sequence[sent_id][seq_id])<<endl;
						file << decoder_output_vocab.get_word(predicted_sequence[sent_id][seq_id])<<" ";	
					}
					file<<endl;
				}
			}
			//Now to print the input and output hidden states
			if (arg_hidden_states_file != ""){
				//Then print the hidden 
				for (int minibatch_index=0; minibatch_index<current_minibatch_size; minibatch_index++){
					if (arg_run_lm == 0){
						//hidden_states_file<<"--ENCODER_STATES--"<<endl;
						//Then first generate the hidden from the encoder
						for (int word_index=0; word_index<max_input_sent_len; word_index++){
							if (testing_input_sequence_cont_sent_data(word_index,minibatch_index) == 1){																		
							hidden_states_file << "input_symbol: "<<encoder_vocab.get_word(testing_input_sent_data(word_index,minibatch_index))<<endl;
							//now printing all the states
							writeStates(hidden_states_file,
										encoder_h_t,
										encoder_c_t,
										encoder_f_t,
										encoder_i_t,
										encoder_o_t,
										minibatch_index,
										word_index);																											
							/*
							for (int hdim=0; hdim < myParam.num_hidden; hdim++){
								hidden_states_file<<input_hiddens[minibatch_index](hdim,word_index)<<" ";
								
							}
							hidden_states_file<<endl;	
							*/
						}						
					}

					//First get the sentence length
						int current_sentence_length=0;
					for (int word_index=0; word_index<max_output_sent_len; word_index++) {
						if (testing_output_sequence_cont_sent_data(word_index,minibatch_index) == 1)
							current_sentence_length++;	
					}
					//cerr<<"current sentence length is "<<current_sentence_length<<endl;
					current_sentence_length--; //this is because the last word in the output sentence is </s> 
					//The first symbol will be '<s>
					hidden_states_file << "input_symbol: <s>" << endl;
					writeStates(hidden_states_file,
								decoder_h_t,
								decoder_c_t,
								decoder_f_t,
								decoder_i_t,
								decoder_o_t,
								minibatch_index,
								0);						
					for (int word_index=1; word_index<current_sentence_length; word_index++){
						//cerr<<"testing_output_sequence_cont_sent_data("<<word_index<<","<<minibatch_index<<") "
						//		<<testing_output_sequence_cont_sent_data(word_index,minibatch_index)<<endl;
		
						
						if (testing_output_sequence_cont_sent_data(word_index,minibatch_index) == 1){																		
						hidden_states_file << "input_symbol: " << decoder_output_vocab.get_word(testing_output_sent_data(word_index,minibatch_index))<<endl;
						writeStates(hidden_states_file,
									decoder_h_t,
									decoder_c_t,
									decoder_f_t,
									decoder_i_t,
									decoder_o_t,
									minibatch_index,
									word_index);		
																													
						/*
						for (int hdim=0; hdim < myParam.num_hidden; hdim++){
							hidden_states_file<<output_hiddens[minibatch_index](hdim,word_index)<<" ";
						}
						hidden_states_file<<endl;	
						*/
					}														
				}
				hidden_states_file<<"<<<<<<<<<NEW SENTENCE>>>>>>>>>"<<endl;
			  }	
			}
		}	

	 }//End minibatch  
	 
	 if (arg_predicted_sequence_file != ""){
		 file.close();
	 }
	 if(arg_hidden_states_file != "")
	 	hidden_states_file.close();
	 cerr << "done." << endl;
	 if(arg_score) {
        //cerr << "Validation log-likelihood: "<< log_likelihood << endl;
        //cerr << "           perplexity:     "<< exp(-log_likelihood/validation_data_size) << endl;
		cerr << "		Test corpus probability          :   " << exp(log_likelihood) << endl; 
	    cerr << "		Testing log-likelihood base e    :   " << log_likelihood << endl;
		cerr << "		Testing log-likelihood base 2    :   " << log_likelihood/log(2.) << endl;
		cerr<<  "		Testing cross entropy in base 2  :   "<< log_likelihood/(log(2.)*total_output_tokens)<< endl;
		cerr << "         		perplexity               :   "<< exp(-log_likelihood/total_output_tokens) << endl;	 	
	 } 	 


	#ifdef USE_CHRONO
	cerr << "Propagation times:";
	for (int i=0; i<timer.size(); i++)
	  cerr << " " << timer.get(i);
	cerr << endl;
	#endif
	
	

	
    //}
    return 0;
}

