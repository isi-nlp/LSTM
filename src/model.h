#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <string>
#include <boost/random/mersenne_twister.hpp>

#include "neuralClasses.h"
#include "Activation_function.h"

namespace nplm
{

class input_model{
public:
		int num_hidden, input_vocab_size, input_embedding_dimension;
	input_model():
		num_hidden(0),
		input_vocab_size(0),
		input_embedding_dimension(0){}
	input_model(int num_hidden, 
				int input_vocab_size, 
				int input_embedding_dimension) :
				num_hidden(num_hidden),
				input_vocab_size(input_vocab_size),
				input_embedding_dimension(input_embedding_dimension){}
				
	virtual void resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden)	= 0;

	virtual void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon) = 0;
	
	virtual void scale(const precision_type scaling_constant) {
		cerr<<"dropout scaling not implemented"<<endl;
	}
	virtual void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							bool norm_clipping,
							precision_type norm_threshold) = 0;
	virtual void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							precision_type grad_scale) {
		cerr<<"write not implemented "<<endl;
		exit(1);
	}							
	virtual void resetGradient() = 0;	
	virtual void write(std::ofstream &file) {
		cerr<<"write not implemented "<<endl;
		exit(1);
	};			
	virtual void read(const string &filename) {
		cerr<<"Read not implemented "<<endl;
		exit(1);
	}
	virtual void readConfig(ifstream config_file) {
		cerr<<"Write config not implemented"<<endl;
		exit(1);
	}
	virtual precision_type getGradSqdNorm() {
		return(0);
	}
};


	//template <class input_model_type>
class model {
public:
    //Input_word_embeddings input_layer, W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
    Linear_layer first_hidden_linear;
    Activation_function first_hidden_activation;
    Linear_layer second_hidden_linear;
    Activation_function second_hidden_activation;
    Output_word_embeddings output_layer;
	//output_model *output;	

    //Linear_layer W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
  	Linear_layer W_h_to_i, W_h_to_f, W_h_to_c, W_h_to_o;
  	Linear_diagonal_layer W_c_to_i, W_c_to_f, W_c_to_o;
    Hidden_layer i_t,f_t,o_t,tanh_c_prime_t;
  	Activation_function tanh_c_t;
	input_model *input;
	//input_model_type * input_model;
    
    activation_function_type activation_function;
    int ngram_size, input_vocab_size, output_vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
    bool premultiplied;

    model(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension,
        bool share_embeddings) 
    {

        resize(ngram_size,
            input_vocab_size,
            output_vocab_size,
            input_embedding_dimension,
            num_hidden,
            output_embedding_dimension);
    }
	int get_hidden() {return num_hidden;}
	
	void set_input(input_model &input){this->input = &input;}
	
    model() : input(NULL),
			ngram_size(1), 
            premultiplied(false),
            activation_function(Rectifier)//,

        {

        }
		
	precision_type getGradSqdNorm();

    void resize(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
		precision_type init_output_bias,
        precision_type init_forget_bias,
        string &parameter_udpate,
        precision_type adagrad_epsilon);

    void set_activation_function(activation_function_type f)
    {
        activation_function = f;
        first_hidden_activation.set_activation_function(f);
        second_hidden_activation.set_activation_function(f);
    }
	
    void set_activation_functions()
    {
		o_t.set_activation_function(Sigmoid);

		f_t.set_activation_function(Sigmoid);	

		i_t.set_activation_function(Sigmoid);		
	
		tanh_c_prime_t.set_activation_function(Tanh);
	
		tanh_c_t.set_activation_function(Tanh);
    }
    //void premultiply();

    // Since the vocabulary is not essential to the model,
    // we need a version with and without a vocabulary.
    // If the number of "extra" data structures like this grows,
    // a better solution is needed

    void read(const std::string &filename);
	//function for getting the start and end words
	void read(const string &filename, int &start_word, int &end_word);
    void read(const std::string &filename, std::vector<std::string> &words);
    void read(const std::string &filename, std::vector<std::string> &input_words, std::vector<std::string> &output_words);
    void write(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> &words);
    void write(const std::string &filename, const std::vector<std::string> &input_words, const std::vector<std::string> &output_words);
	void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
				  		precision_type momentum,
						precision_type L2_reg,
						bool norm_clipping,
						precision_type norm_threshold);
	void updateParams(precision_type learning_rate,
	 					int current_minibatch_size,
				  		precision_type momentum,
						precision_type L2_reg,
						precision_type grad_scale);						
	void resetGradient();			
	void scale(precision_type scaling_constant);
 private:
    void readConfig(std::ifstream &config_file);
    void readConfig(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> *input_pwords, const std::vector<std::string> *output_pwords);
};




class google_input_model : public input_model {

public:
    Input_word_embeddings input_layer;
    //Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor> input_embedding_matrix;
	Linear_layer W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
 
    int num_hidden, input_vocab_size, input_embedding_dimension;

    google_input_model(int num_hidden,
		int input_vocab_size,
        int input_embedding_dimension):input_model(num_hidden,input_vocab_size,input_embedding_dimension)  
    {
          //input_embedding_matrix = Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>();
          //input_layer.set_W(&input_embedding_matrix);
    }
	
    google_input_model() : 
			input_model() //,
            //input_embedding_matrix(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>())
        {
          //input_layer.set_W(&input_embedding_matrix);  
        }
	
    void resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon);	
		
	void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							bool norm_clipping,
							precision_type norm_threshold);
	void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							precision_type grad_scale);									
	void resetGradient();
	void scale(const precision_type scaling_constant);
	void write(ofstream &file);
	void read(const string &filename);
	void readConfig(ifstream &config_file);
	precision_type getGradSqdNorm();
    void readEmbeddingsFromFile(const std::string &file, vocabulary &vocab){
    	input_layer.readEmbeddingsFromFile(file, vocab);
    }
};

class hidden_to_hidden_input_model : public input_model {

public:

	Linear_layer W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
 
    int num_hidden, input_vocab_size, input_embedding_dimension;

    hidden_to_hidden_input_model(int num_hidden,
		int input_vocab_size,
        int input_embedding_dimension):input_model(num_hidden,input_vocab_size,input_embedding_dimension)  
    {

    }
	
    hidden_to_hidden_input_model() : 
			input_model()
        {
          //input_layer.set_W(&input_embedding_matrix);  
        }

    void resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon);	
		
	void updateParams(precision_type learning_rate,
		 					int current_minibatch_size,
					  		precision_type momentum,
							precision_type L2_reg,
							bool norm_clipping,
							precision_type norm_threshold);
					
	void resetGradient();
};


class standard_input_model : public input_model{

public:
    Input_word_embeddings W_x_to_i, W_x_to_f, W_x_to_c, W_x_to_o;
    Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>  W_x_to_i_embedding_matrix, 
	  W_x_to_f_embedding_matrix, 
	  W_x_to_c_embedding_matrix, 
	  W_x_to_o_embedding_matrix; 
 
    //int num_hidden, input_vocab_size, input_embedding_dimension;

    standard_input_model(int num_hidden,
		int input_vocab_size,
        int input_embedding_dimension) :input_model(num_hidden,input_vocab_size,input_embedding_dimension) 
    {
		/*
	  W_x_to_i_embedding_matrix = Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>();
	  W_x_to_f_embedding_matrix = Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>();
	  W_x_to_o_embedding_matrix = Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>();
	  W_x_to_c_embedding_matrix = Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>();
	  W_x_to_i.set_W(&W_x_to_i_embedding_matrix);
	  W_x_to_f.set_W(&W_x_to_f_embedding_matrix);
	  W_x_to_c.set_W(&W_x_to_c_embedding_matrix);
	  W_x_to_o.set_W(&W_x_to_o_embedding_matrix);
		*/
    }
	
    standard_input_model() : 
		input_model()//,
		/*
		W_x_to_i_embedding_matrix(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>()),
		W_x_to_f_embedding_matrix(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>()),
		W_x_to_o_embedding_matrix(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>()),
		W_x_to_c_embedding_matrix(Matrix<precision_type,Dynamic,Dynamic,Eigen::RowMajor>())
		*/
        {
			/*
  		  W_x_to_i.set_W(&W_x_to_i_embedding_matrix);
  		  W_x_to_f.set_W(&W_x_to_f_embedding_matrix);
  		  W_x_to_c.set_W(&W_x_to_c_embedding_matrix);
  		  W_x_to_o.set_W(&W_x_to_o_embedding_matrix);	
			*/ 
        }

    void resize(int input_vocab_size,
    int input_embedding_dimension,
    int num_hidden);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        precision_type init_range,
        string &parameter_udpate,
        precision_type adagrad_epsilon);
			
		void updateParams(precision_type learning_rate,
			 					int current_minibatch_size,
						  		precision_type momentum,
								precision_type L2_reg,
								bool norm_clipping,
								precision_type norm_threshold);
	   void resetGradient(){}
};

} //namespace nplm

//#include "model.ipp"
#endif
