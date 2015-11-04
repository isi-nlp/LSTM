#ifndef SOFTMAXLOSS_H
#define SOFTMAXLOSS_H

#include <Eigen/Dense>
#include "multinomial.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

///// Softmax layer plus log-loss function.

enum loss_function_type { LogLoss, NCELoss, InvalidLoss };

inline loss_function_type string_to_loss_function (const std::string &s)
{
    if (s == "log")
        return LogLoss;
    else if (s == "nce")
        return NCELoss;
    else
        return InvalidLoss;
}

inline std::string loss_function_to_string (loss_function_type f)
{
    if (f == LogLoss)
        return "log";
    else if (f == NCELoss)
        return "nce";
    else {
        std::cerr<< "InvalidLoss"<<std::endl;
		exit(1);
	}
}

/// Note: Outputs log-probabilities.

struct SoftmaxLogLoss
{
    template <typename DerivedI, typename DerivedW, typename DerivedO>
    void fProp(const MatrixBase<DerivedI> &input, 
			const MatrixBase<DerivedW> &output_words, 
			const MatrixBase<DerivedO> &output_const, 
			precision_type &loss)
    {
		//std::cerr<<"output words are "<<output_words<<std::endl;
		//std::cerr<<"output const is "<<output_const<<std::endl;
		//std::cerr<<"input is "<<input<<std::endl;
	    UNCONST(DerivedO, output_const, output);
		//std::cerr<<"input is "<<input<<std::endl;

		//getchar();

		double log_likelihood = 0.0;
	
	    #pragma omp parallel for reduction(+:log_likelihood)
		for (int train_id = 0; train_id < input.cols(); train_id++)
		{
			//std::cerr<<"output word "<<output_words(train_id)<<std::endl;
			//If the output word is negative, that means there was no sample
			if (output_words(train_id) == -1){
				//std::cerr<<"word is -1"<<std::endl;
				continue;
			}
		    double normalization = logsum(input.col(train_id));
		    output.col(train_id).array() = input.col(train_id).array() - normalization;
			//std::cerr<<"normalization is"<<normalization<<std::endl;
		    log_likelihood += double(output(output_words(train_id), train_id));
		}
		//std::cerr<<"output is "<<output<<std::endl;
		//getchar();
		loss = log_likelihood;
    }

    template <typename DerivedI, typename DerivedW, typename DerivedO>
    void computeProbs(const MatrixBase<DerivedI> &input, 
			const MatrixBase<DerivedW> &output_words, 
			const MatrixBase<DerivedO> &output_const)
    {
		//std::cerr<<"output words are "<<output_words<<std::endl;
		//std::cerr<<"output const is "<<output_const<<std::endl;
	    UNCONST(DerivedO, output_const, output);
		//std::cerr<<"input is "<<input<<std::endl;

		//getchar();

		double log_likelihood = 0.0;
	
	    #pragma omp parallel for
		for (int train_id = 0; train_id < input.cols(); train_id++)
		{
			//std::cerr<<"output word "<<output_words(train_id)<<std::endl;
			//If the output word is negative, that means there was no sample
			if (output_words(train_id) == -1){
				//std::cerr<<"word is -1"<<std::endl;
				continue;
			}
		    double normalization = logsum(input.col(train_id));
		    output.col(train_id).array() = input.col(train_id).array() - normalization;
			//std::cerr<<"normalization is"<<normalization<<std::endl;
		}

    }
	
    template <typename DerivedW, typename DerivedO, typename DerivedI>
    void bProp(const MatrixBase<DerivedW> &output_words, const MatrixBase<DerivedO> &output, const MatrixBase<DerivedI> &grad_input_const)
    {
        UNCONST(DerivedI, grad_input_const, grad_input);
        grad_input.setZero();
        #pragma omp parallel for
		for (int train_id = 0; train_id < output.cols(); train_id++)
		{
			//If the output word is -1, there is no gradient
			if (output_words(train_id) == -1) {
				continue;
			}
		    grad_input(output_words(train_id), train_id) += 1.;
		    //grad_input.col(train_id) -= output.col(train_id).array().exp().matrix();
			grad_input.col(train_id) -= output.col(train_id).array().exp().matrix();
			
		}
		//std::cerr<<"grad input is "<<grad_input<<std::endl;
    }
};

///// Softmax layer plus NCE loss function.

///// Note: Outputs probabilities.

///// Note: Unlike SoftmaxLogLoss, does not compute *or* apply precomputed
///// normalizations. Currently the caller is expected to do normalization.

//template <typename Multinomial>
class SoftmaxNCELoss
{
    const multinomial<data_size_t> *unigram;

public:
	SoftmaxNCELoss()
		:unigram(NULL)
	{
	}
    SoftmaxNCELoss( const multinomial<data_size_t> *unigram) 
		: unigram(unigram)
    {
		//this->unigram = &unigram;
    }
	void set_unigram( const multinomial<data_size_t> *unigram) {
			this->unigram = unigram;
	}
    template <typename DerivedI, typename DerivedW, typename DerivedO>
    void fProp(const MatrixBase<DerivedI> &scores, 
	       const MatrixBase<DerivedW> &minibatch_samples,
	       const MatrixBase<DerivedO> &output_const, 
		   precision_type &loss,
		   precision_type &fixed_partition_function)
    {
        UNCONST(DerivedO, output_const, output);
		//UNCONST(DerivedW, const_minibatch_samples, minibatch_samples);
		precision_type log_likelihood = 0.0;
		int num_noise_samples = minibatch_samples.rows()-1;
		//std::cerr<<"num noise samples are "<<num_noise_samples<<std::endl;
		precision_type log_num_noise_samples = std::log(num_noise_samples);
		//std::cerr<<"minibatch samples are "<<minibatch_samples<<std::endl;
		//getchar();
        //#pragma omp parallel for reduction(+:log_likelihood) schedule(static)
		for (int train_id = 0; train_id < scores.cols(); train_id++)
		{
			for (int sample_id = 0;sample_id < minibatch_samples.rows(); sample_id++) {
				int sample = minibatch_samples(sample_id, train_id);
				//std::cerr<<"sample id "<<sample_id<<", train_id "<<train_id<<std::endl;
				//std::cerr<<" minibatch sample "<<sample<<" \nand prob is "<<
				//	log_num_noise_samples + unigram->logprob(sample)<<std::endl;	
			}		
			precision_type local_log_likelihood = 0;

			//If the output word is -1, continue
			if (minibatch_samples(0, train_id) == -1) {
				output.col(train_id).setZero();
				output(0,train_id) = 1; //Setting this to 1 because it will be set to 0 in the backprop phase, which implies 0 gradient
				//std::cerr<<"item was -1"<<std::endl;
				continue;
			} else {
			    for (int sample_id = 0;sample_id < minibatch_samples.rows(); sample_id++)
			    {
					//cerr<<" for sample "<<sample<<" log_num_noise_samples + unigram->logprob(sample) "<<
					//	sample<<log_num_noise_samples + unigram->logprob(sample)<<endl;
			        int sample = minibatch_samples(sample_id, train_id);
					// To avoid zero or infinite probabilities,
					// never take exp of score without normalizing first,
					// even if it's a little slower...
					precision_type score = scores(sample_id, train_id)-fixed_partition_function;
					precision_type score_noise = log_num_noise_samples + unigram->logprob(sample);
					precision_type z = logadd(double(score), double(score_noise));
					precision_type logprob = score - z;
					precision_type logprob_noise = score_noise - z;
					output(sample_id, train_id) = std::exp(double(logprob));
					local_log_likelihood += sample_id == 0 ? logprob : logprob_noise;
			    }
			}
			log_likelihood += local_log_likelihood;
		}
		loss = log_likelihood;
		/*
		Matrix<precision_type,Dynamic,Dynamic> only_p_trues = output;
		only_p_trues.bottomRows(num_noise_samples)  = Matrix<precision_type,Dynamic,Dynamic>::Ones(num_noise_samples,only_p_trues.cols())
						-only_p_trues.bottomRows(num_noise_samples);
		std::cerr<<"Ptrues for everything are "<<only_p_trues<<std::endl;
		getchar();
		*/
		
	 }

    template <typename DerivedO, typename DerivedI>
    void bProp(const MatrixBase<DerivedO> &probs, const MatrixBase<DerivedI> &output_const)
    {
        UNCONST(DerivedI, output_const, output);
        #pragma omp parallel for schedule(static)
		for (int train_id = 0; train_id < probs.cols(); train_id++)
		{
		    output.col(train_id) = -probs.col(train_id);
		    output(0, train_id) += 1.0;
		}
		//std::cerr<<"gradient from output layer is "<<output<<std::endl;
		//getchar();
    }
};

} // namespace nplm

#endif
