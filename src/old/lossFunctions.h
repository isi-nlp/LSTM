#include <iostream>
#include <list>
#include <ctime>
#include <cstdio>

#include "param.h"
#include "neuralClasses.h"
//#include "graphClasses.h"
//#include "util.h"
//#include "RBMDahlFunctions.h"

//#include<tr1/random>
#include <time.h>
//#include <chrono>
//#include <random>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdio.h>
#include <iomanip>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <math.h>
#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
typedef Matrix<int,Dynamic,Dynamic> IntDynamicMatrix;
typedef Matrix<double,Dynamic,Dynamic> RealDynamicMatrix;
typedef  Matrix<double,Dynamic,1> RealColumn;
typedef  Matrix<int,Dynamic,1> IntColumn;


void typeALoss(param &myParam, vector< IntColumn > &shuffled_training_data,RealDynamicMatrix &minibatch_predicted_embeddings,
             RealColumn &minibatch_positive_weights,RealDynamicMatrix &minibatch_negative_weights,IntDynamicMatrix &minibatch_negative_samples,
              Output_word_embeddings &D_prime,vector<context_node > context_nodes,vector<vector<double> > & q_vector, 
              vector<vector<int> > &J_vector,vector<vector<double> > &unigram_probs_vector,
              vector<uniform_real_distribution<> >& unif_real_vector,vector<mt19937> & eng_real_vector,
              vector<uniform_int_distribution<> > & unif_int_vector,vector<mt19937> & eng_int_vector,int minibatch_start_index)
{
    int thread_id = 0;
    for (int train_id = 0;train_id < myParam.minibatch_size;train_id++)
    {
        int output_word = shuffled_training_data[myParam.ngram_size-1](minibatch_start_index+train_id);
        Matrix<double,1,Dynamic> predicted_embedding;
        predicted_embedding.setZero(myParam.embedding_dimension);
        //getting the predicted context vector
        for (int word = 0;word<myParam.ngram_size-1;word++)
        {
            predicted_embedding += context_nodes[word].fProp_matrix.col(train_id);
        }
        minibatch_predicted_embeddings.col(train_id) = predicted_embedding;
        double score = D_prime.W.row(output_word).dot(predicted_embedding) + D_prime.b(output_word);
        double unnorm_positive_prob = exp(score);
        minibatch_positive_weights(train_id) = myParam.num_noise_samples*unigram_probs_vector[thread_id][output_word]/
                                               (unnorm_positive_prob + myParam.num_noise_samples * unigram_probs_vector[thread_id][output_word]) ;
        cout<<" the positive score is "<<score<<endl;
        cout<<" the positive weight is "<<minibatch_positive_weights(train_id)<<endl;
        cout<<" unnorm positive prob is "<<unnorm_positive_prob<<endl;
        cout<<" unigram prob is "<<unigram_probs_vector[thread_id][output_word]<<endl;
        
        int thread_id = 0; //for now, this is a proxy. When I'm multithreading this code, the thread ID will change
        cout<<"generating "<<myParam.num_noise_samples<<" noise samples for the training example"<<endl;
        //first generating noise samples
        
        for (int sample_id = 0;sample_id <myParam.num_noise_samples;sample_id++)
        {

            int mixture_component = unif_int_vector[thread_id](eng_int_vector[thread_id]);
            //cout<<"mixture component was "<<mixture_component<<endl;
            double p = unif_real_vector[thread_id](eng_real_vector[thread_id]);
            int sample ;
            //cout<<"computing sample"<<endl;
            //cout<<"remaining bernoulli item is "<<J_vector[thread_id][mixture_component]<<endl;
            if (q_vector[thread_id][mixture_component] >= p)
            {
                //cout<<"mixture accepted"<<endl;
                sample = mixture_component;
            }
            else
            {
                //cout<<"J accepted "<<endl;
                sample = J_vector[thread_id][mixture_component];
            }
            cout<<"the sample was "<<sample<<endl;
            assert (sample >= 0);
            minibatch_negative_samples(train_id,sample_id) = sample;
            double negative_score = D_prime.W.row(sample).dot(predicted_embedding) + D_prime.b(sample);
            double negative_unnorm_positive_prob = exp(score);
            minibatch_negative_weights(train_id,sample_id) = negative_unnorm_positive_prob/
                                                            (negative_unnorm_positive_prob + myParam.num_noise_samples * unigram_probs_vector[thread_id][sample]);
            cout<<"the negative score was "<<negative_score<<endl;
            cout<<"the unnorm negative prob was "<<negative_unnorm_positive_prob<<endl;
            
        }
        //getting the negative gradient using noise samples for the ouput layer
        

        

    }

}

//void computePredictedEmbeddings()

