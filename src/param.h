#pragma once

#include <string>

namespace nplm
{

struct param 
{
    std::string train_file;
    std::string validation_file;
    std::string test_file;

    std::string model_file;

    std::string unigram_probs_file;
    std::string words_file;
    std::string input_words_file;
    std::string output_words_file;
    std::string model_prefix;
	std::string input_sent_file;
	std::string output_sent_file;
	std::string input_validation_sent_file;
	std::string output_validation_sent_file;
	std::string training_sequence_cont_file;
	std::string testing_sequence_cont_file;
	std::string validation_sequence_cont_file;

    int ngram_size;
    int vocab_size;
    int input_vocab_size;
    int output_vocab_size;
    int num_hidden;
    int embedding_dimension;
    int input_embedding_dimension;
    int output_embedding_dimension;
    std::string activation_function;
    std::string loss_function;
    std::string parameter_update;

    int minibatch_size;
    int validation_minibatch_size;
    int num_epochs;
    double learning_rate;
    double conditioning_constant;
    double decay;
    double adagrad_epsilon;
    bool init_normal;
    double init_range;
	double init_forget;
	bool norm_clipping;
	bool gradient_check;
	bool restart_states;
	
    int num_noise_samples;

    bool use_momentum;
    double initial_momentum;
    double final_momentum;

    double L2_reg;
	double norm_threshold;

    bool normalization;
    double normalization_init;

    int num_threads;
  
    bool share_embeddings;

};

} // namespace nplm

