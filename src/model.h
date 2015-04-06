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

class model {
public:
    Input_word_embeddings input_layer;
	Input_word_embeddings rhs_rule_input_layer,
						  Td_input_layer;
    Linear_layer first_hidden_linear;
    Activation_function first_hidden_activation;
    Linear_layer second_hidden_linear;
    Activation_function second_hidden_activation;
	Activation_function D1_hidden_activation,
						Td1_hidden_activation,
						d1_hidden_activation; 
    Output_word_embeddings output_layer,
			rhs_rule_output_layer,
			Td_output_layer;
	Output_word_embeddings d_output_layer;
			
	Linear_layer rhs_rule_linear_layer,
				 Td_linear_layer;
				 
    Matrix<double,Dynamic,Dynamic,Eigen::RowMajor> output_embedding_matrix,
		rhs_rule_embedding_matrix,
		Td_embedding_matrix,
		d_embedding_matrix,
      input_embedding_matrix,
      input_and_output_embedding_matrix;
    
    activation_function_type activation_function;
    int ngram_size, 
		input_vocab_size, 
		output_vocab_size, 
		input_embedding_dimension, 
		num_hidden, 
		output_embedding_dimension,
		rhs_rule_vocab_size,
		Td_vocab_size,
		d_vocab_size;
    bool premultiplied;


    model(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension,
        bool share_embeddings,
		int rhs_rule_vocab_size,
		int Td_vocab_size) 
    {
        if (share_embeddings){
          input_and_output_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
          input_layer.set_W(&input_and_output_embedding_matrix);
          output_layer.set_W(&input_and_output_embedding_matrix);
        }
        else {
          input_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
          output_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
		  rhs_rule_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
		  Td_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
		  d_embedding_matrix = Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>();
		  
          input_layer.set_W(&input_embedding_matrix);
          output_layer.set_W(&output_embedding_matrix);
		  rhs_rule_output_layer.set_W(&rhs_rule_embedding_matrix);
		  Td_output_layer.set_W(&Td_embedding_matrix);
		  d_output_layer.set_W(&d_embedding_matrix);
		  rhs_rule_input_layer.set_W(&rhs_rule_embedding_matrix);
		  Td_input_layer.set_W(&Td_embedding_matrix);
        }
        resize(ngram_size,
            input_vocab_size,
            output_vocab_size,
            input_embedding_dimension,
            num_hidden,
            output_embedding_dimension,
			rhs_rule_vocab_size,
			Td_vocab_size);
    }
    model() : ngram_size(1), 
            premultiplied(false),
            activation_function(Rectifier),
            output_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
            input_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
			d_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
			Td_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>()),
			rhs_rule_embedding_matrix(Matrix<double,Dynamic,Dynamic,Eigen::RowMajor>())
        {
          output_layer.set_W(&output_embedding_matrix);
          input_layer.set_W(&input_embedding_matrix);
		  d_output_layer.set_W(&d_embedding_matrix);
		  Td_output_layer.set_W(&Td_embedding_matrix);
		  Td_input_layer.set_W(&Td_embedding_matrix);
		  rhs_rule_output_layer.set_W(&rhs_rule_embedding_matrix);
		  rhs_rule_input_layer.set_W(&rhs_rule_embedding_matrix);
        }

    void resize(int ngram_size,
        int input_vocab_size,
        int d_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension,
		int rhs_rule_vocab_size,
		int Td_vocab_size);

    void initialize(boost::random::mt19937 &init_engine,
        bool init_normal,
        double init_range,
        double init_d_bias,
		double init_Td_bias,
		double init_rhs_rule_bias,
        string &parameter_udpate,
        double adagrad_epsilon);

    void set_activation_function(activation_function_type f)
    {
        activation_function = f;
        first_hidden_activation.set_activation_function(f);
        second_hidden_activation.set_activation_function(f);
		D1_hidden_activation.set_activation_function(f);
		Td1_hidden_activation.set_activation_function(f);
		d1_hidden_activation.set_activation_function(f);
    }

    void premultiply();

    // Since the vocabulary is not essential to the model,
    // we need a version with and without a vocabulary.
    // If the number of "extra" data structures like this grows,
    // a better solution is needed

    void read(const std::string &filename);
    void read(const std::string &filename, std::vector<std::string> &words);
    void read(const std::string &filename, std::vector<std::string> &input_words, std::vector<std::string> &output_words);
    void write(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> &words);
    void write(const std::string &filename, const std::vector<std::string> &input_words, const std::vector<std::string> &output_words);

 private:
    void readConfig(std::ifstream &config_file);
    void readConfig(const std::string &filename);
    void write(const std::string &filename, const std::vector<std::string> *input_pwords, const std::vector<std::string> *output_pwords);
};

} //namespace nplm

#endif
