#include <iostream>
#include <list>
#include <ctime>
#include <cstdio>

#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

#include "param.h"
#include "neuralClasses.h"
//#include "graphClasses.h"
#include "util.h"
//#include "RBMDahlFunctions.h"
#include "log_add.h"
#include <cmath>
#include <stdlib.h>

typedef Node <Word_embeddings> word_node;
typedef Node <Context_matrix> context_node;
typedef Node <Hidden_layer> hidden_node;

//#include "lossFunctions.h"


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
#include "maybe_omp.h"
#include <math.h>
#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include <stdlib.h>

typedef boost::unordered_map<vector<int>, double> vector_map;
typedef boost::unordered_map<int,vector_map > thread_vector_map;
typedef Eigen::Matrix<double,Dynamic,Dynamic> RealMatrix;

using namespace std;
using namespace Eigen;
using namespace boost::random;

void inline fPropGradCheck(param & ,int ,int ,vector<word_node > &,vector<context_node > &,hidden_node &,context_node &,vector<Matrix<int,Dynamic,1> >&);
double computeLossFunction(param & ,int ,int ,vector<word_node > &,
            vector<context_node > &,hidden_node &,context_node & ,
            vector<Matrix<int,Dynamic,1> >&,vector_map & ,Matrix<double,Dynamic,Dynamic> & ,
            Matrix<int,Dynamic,Dynamic> & ,vector<vector<double> > &,Output_word_embeddings & );
void initZero(param & ,vector<word_node > &,vector<context_node > &,hidden_node &,
              context_node & );


void gradientChecking(param & myParam,int minibatch_start_index,int current_minibatch_size,vector<word_node > &word_nodes,
            vector<context_node > &context_nodes,hidden_node &hidden_layer_node,context_node & hidden_layer_to_output_node,
            vector<Matrix<int,Dynamic,1> >&shuffled_training_data,vector_map &c_h,vector<uniform_real_distribution<> >& unif_real_vector,
            vector<mt19937> & eng_real_vector,vector<uniform_int_distribution<> > & unif_int_vector,vector<mt19937> & eng_int_vector,
            vector<vector<double> > &unigram_probs_vector,vector<vector<double> > & q_vector,vector<vector<int> >&J_vector,Output_word_embeddings & D_prime)
{ 
    double delta_perturb = 0.000005;
    std::setprecision(20);
    //creating the gradient matrices
    RealMatrix gradient_input_W;
    RealMatrix gradient_output_W;
    Matrix<double,Dynamic,1>gradient_output_b;
    Matrix<double,Dynamic,1>gradient_h_bias;
    RealMatrix gradient_hidden_to_output_matrix;
    vector<RealMatrix> gradients_context_matrix;

    int ngram_size = myParam.ngram_size;
    int n_hidden = myParam.n_hidden;
    int n_vocab = myParam.n_vocab;
    int embedding_dimension = myParam.embedding_dimension;
    int minibatch_size = myParam.minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;
    double normalization_init = myParam.normalization_init;

    gradient_input_W.setZero(myParam.n_vocab,myParam.embedding_dimension);
    gradient_output_W.setZero(myParam.n_vocab,myParam.embedding_dimension);
    gradient_output_b.setZero(myParam.n_vocab);
    gradient_h_bias.setZero(myParam.n_hidden);
    gradient_hidden_to_output_matrix.setZero(myParam.embedding_dimension,myParam.n_hidden);
    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        RealMatrix context_matrix_gradient;
        context_matrix_gradient.setZero(myParam.n_hidden,myParam.embedding_dimension);
        gradients_context_matrix.push_back(context_matrix_gradient);
    }


    ///////////////////FORWARD PROPAGATION//////////////////////////
    ////////////////////////////////////////////////////////////////
    fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);

    /////////////////COMPUTING THE NCE LOSS FUNCTION//////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    //computing the loss function 
    //now computing the loss function
    //Matrix<double,Dynamic,Dynamic> output_gradient;
    Matrix<double,Dynamic,Dynamic> minibatch_predicted_embeddings(myParam.embedding_dimension,current_minibatch_size);
    Matrix<double,Dynamic,1> minibatch_positive_weights(current_minibatch_size);
    Matrix<double,Dynamic,Dynamic> minibatch_negative_weights(current_minibatch_size,myParam.num_noise_samples);
    Matrix<int,Dynamic,Dynamic> minibatch_negative_samples(current_minibatch_size,myParam.num_noise_samples);


    
    //int thread_id = 0; //for now, this is a proxy. When I'm multithreading this code, the thread ID will change
    //creating the unordered map for each thread
    //thread_vector_map c_h_gradient_vector;
    vector<vector_map> c_h_gradient_vector;
    vector_map c_h_gradient;
    for (int thread_id =0 ;thread_id<myParam.n_threads;thread_id++)
    {
        vector_map temp;
        c_h_gradient_vector.push_back(temp);
    }
    clock_t t;
    t = clock();
    //c_h_gradient_vector += minibatch_positive_weights(train_id)
    //parallelizing the creation with multithreading
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    cout<<"staring the fprop"<<endl;
    #pragma omp parallel firstprivate(current_minibatch_size,minibatch_start_index,ngram_size,n_vocab,embedding_dimension, \
                            num_noise_samples,normalization_init)
    {
    #pragma omp for //schedule(dynamic)
    for (int train_id = 0;train_id < minibatch_size;train_id++)
    {

        int thread_id = omp_get_thread_num();
        int output_word = shuffled_training_data[ngram_size-1](minibatch_start_index+train_id);
        //cout<<"output word is "<<output_word<<endl;
        Matrix<double,Dynamic,1> predicted_embedding = hidden_layer_to_output_node.fProp_matrix.col(train_id);

        vector<int> context;//(ngram_size-1);

        //creating the context
        for (int word = 0;word<ngram_size-1;word++)
        {

            //cout<<"train id is "<<train_id<<endl;
            //cout<<"minibatch start index is "<<minibatch_start_index<<endl;
            context.push_back(shuffled_training_data[word](minibatch_start_index+train_id));
            cout<<"word "<<word<<" in context is "<<shuffled_training_data[word](minibatch_start_index+train_id)<<endl;
            //context.push_back((*thread_data_col_locations[thread_id][word])(minibatch_start_index+train_id));
        }

        double log_inv_normalization_const_h = 0.;
        //getting a normalization constant and making it threasafe
        
        //this region does not need to be critical because its just a read
        log_inv_normalization_const_h = c_h[context];

        double inv_normalization_const_h = exp(log_inv_normalization_const_h);
        //cout<<"The normalization constant is "<<inv_normalization_const_h<<endl;
        //setting the gradient for that context to 0;
        //double c_h_gradient_vector = 0.0;

        minibatch_predicted_embeddings.col(train_id) = predicted_embedding;
        double score = D_prime.W.row(output_word).dot(predicted_embedding) + D_prime.b(output_word);
        double unnorm_positive_prob = exp(score);
        minibatch_positive_weights(train_id) = num_noise_samples*unigram_probs_vector[thread_id][output_word]/
                                               (unnorm_positive_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][output_word]) ;

        
        if (c_h_gradient_vector[thread_id].find(context) == c_h_gradient_vector[thread_id].end())
        {
            c_h_gradient_vector[thread_id][context] = minibatch_positive_weights(train_id);
        }
        else
        {
            //cout<<"we got a repeat!"<<endl;
            c_h_gradient_vector[thread_id][context] += minibatch_positive_weights(train_id);
        }

       
        ///COMPUTING NOISE SAMPLES///
        for (int sample_id = 0;sample_id <num_noise_samples;sample_id++)
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

            //vector<int> context(ngram_size-1);

            //cout<<"the sample was "<<sample<<endl;
            assert (sample >= 0);
            minibatch_negative_samples(train_id,sample_id) = sample;
            double negative_score = D_prime.W.row(sample).dot(predicted_embedding) + D_prime.b(sample);
            double negative_unnorm_prob = exp(negative_score);
            minibatch_negative_weights(train_id,sample_id) = negative_unnorm_prob*inv_normalization_const_h/
                                                            (negative_unnorm_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][sample]);
            c_h_gradient_vector[thread_id][context] -= minibatch_negative_weights(train_id,sample_id);
        }

    }
    }
    #pragma omp barrier

    /////////////////////////////////UPDATING GRADIENTS AND DOING BACKPROPAGATION/////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////
    
    //t = clock();
    //updating the normalization constants
    for (int thread_id=0;thread_id<myParam.n_threads;thread_id++)
    {
        vector_map::iterator it;
        for (it = c_h_gradient_vector[thread_id].begin();it != c_h_gradient_vector[thread_id].end();it++)
        {
            if (c_h_gradient.find((*it).first) == c_h_gradient.end())
            {
                c_h_gradient[(*it).first] = (*it).second;
            }
            else
            {
                c_h_gradient[(*it).first] += (*it).second;
            }

        }
    }

    //cout<<"the time taken to update normalization constants was "<<clock()-t<<endl;
    //t = clock();
    //first comput the backprop gradient
    Matrix<double,Dynamic,Dynamic> context_bProp_matrix;//(myParam.embedding_dimension,current_minibatch_size);
    context_bProp_matrix.setZero(myParam.embedding_dimension,current_minibatch_size);
    D_prime.bProp(shuffled_training_data[myParam.ngram_size-1],minibatch_positive_weights,
                    minibatch_negative_samples,minibatch_negative_weights,
                    context_bProp_matrix,minibatch_start_index,current_minibatch_size,myParam.num_noise_samples);
    //cout<<"the time taken to do bprop on the output layer was "<<clock()-t<<endl;
    //now then update the parameters
    //t = clock();
    D_prime.computeGradientCheck(minibatch_predicted_embeddings,shuffled_training_data[myParam.ngram_size-1],minibatch_positive_weights,
                    minibatch_negative_samples,minibatch_negative_weights,
                    minibatch_start_index,current_minibatch_size,myParam.num_noise_samples,gradient_output_W,gradient_output_b);

    //cout<<"the time taken to compute the gradient on the output layer was "<<clock()-t<<endl;
    //now doing backprop on hidden layer to output matrix
    hidden_layer_to_output_node.param->bProp(context_bProp_matrix,hidden_layer_to_output_node.bProp_matrix);
    hidden_layer_to_output_node.param->computeGradientCheckOmp(context_bProp_matrix,hidden_layer_node.fProp_matrix,gradient_hidden_to_output_matrix);

    //now doing backprop on the hidden node
    hidden_layer_node.param->bPropTanh(hidden_layer_to_output_node.bProp_matrix,hidden_layer_node.bProp_matrix,hidden_layer_node.fProp_matrix);
    hidden_layer_node.param->computeGradientCheckTanh(hidden_layer_to_output_node.bProp_matrix,hidden_layer_node.fProp_matrix,gradient_h_bias);
    
    //now doing backprop on the context matrices
    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        context_nodes[word].param->bProp(hidden_layer_node.bProp_matrix,context_nodes[word].bProp_matrix);
        //updating the context weights
        context_nodes[word].param->computeGradientCheckOmp(hidden_layer_node.bProp_matrix,word_nodes[word].fProp_matrix,gradients_context_matrix[word]);
    }
    //doing backprop on the word embeddings
    for (int word = 0;word < myParam.ngram_size-1;word++)
    {
        //cout<<"the backprop matrix from the context "<<word<<" before doing word updates is "<<context_nodes[word].bProp_matrix<<endl;
        //getchar();

        word_nodes[word].param->computeGradientCheck(context_nodes[word].bProp_matrix,shuffled_training_data[word],minibatch_start_index,
                                                current_minibatch_size,gradient_input_W);
    }
    //compute the NCE LOSS FUNTION
    double current_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                            hidden_layer_to_output_node,shuffled_training_data,c_h,minibatch_predicted_embeddings,minibatch_negative_samples,
                            unigram_probs_vector,D_prime);
    cout<<"the current nce loss is "<<setprecision(10)<<current_nce_loss<<endl;
    //for all the nodes in the graph, I have to set the bprop and fprop matrices to zero
    /*
    hidden_layer_node.bProp_matrix.setZero();
    hidden_layer_node.fProp_matrix.setZero();
    hidden_layer_to_output_node.fProp_matrix.setZero();
    hidden_layer_to_output_node.bProp_matrix.setZero();

    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        context_nodes[word].fProp_matrix.setZero();
        context_nodes[word].bProp_matrix.setZero();
        word_nodes[word].fProp_matrix.setZero();
        //word_nodes[word].bProp_matrix.setZero();
    }
    */
    //initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
    //now that we have the gradients, we check our gradients using finite differences
    
    ////COMPUTING THE LOSS FUNCTION////////////////////////////
    //randomly pick up some parameters whose gradient you want to inspect 
    //first pick some random examples from the minibatch
    //srand (time(NULL));
    //cout<<"the current minibatch size is "<<current_minibatch_size<<endl;
    //cout<<"max is "<<min(4,current_minibatch_size)<<endl;
    getchar();
    for (int example = 0;example <min(4,current_minibatch_size) ;example++)
    {
        //checking the gradient of the normalization constant
        
        cout<<"the example is "<<example<<endl;
        vector<int> context;//(ngram_size-1);

        //creating the context
        for (int word = 0;word<ngram_size-1;word++)
        {
            context.push_back(shuffled_training_data[word](minibatch_start_index+example));
            cout<<"context word "<<word<<" is "<<shuffled_training_data[word](minibatch_start_index+example)<<endl;
            //context.push_back((*thread_data_col_locations[thread_id][word])(minibatch_start_index+train_id));
        }
        c_h[context] +=delta_perturb;
        double perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                            hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                            unigram_probs_vector,D_prime);
        double finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
        double difference = finite_difference_gradient - c_h_gradient[context];
        cout<<"the original gradient is "<<c_h_gradient[context]<<endl;
        cout<<"the finite difference gradient is "<<finite_difference_gradient<<endl; 
        cout<<"the ratio is "<<c_h_gradient[context]/finite_difference_gradient<<endl;
        cout<<"the difference for c_h was "<<abs(difference)<<endl;
        c_h[context] -= delta_perturb;
        getchar();
        //checking the gradient of the hidden layer to output node context matrix
        initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
        int row_perturb_dimension = rand() % n_hidden;
        cout<<"the row perturb dimension was "<<row_perturb_dimension<<endl;
        int col_perturb_dimension = rand() % embedding_dimension;
        cout<<"the col perturb dimension was "<<col_perturb_dimension<<endl;

        //first perturb
        //cout<<"before perturbation the dimension was "<< context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension)<<endl;
        hidden_layer_to_output_node.param->U(row_perturb_dimension,col_perturb_dimension) += delta_perturb;
        //cout<<"after perturbation the dimension was "<< context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension)<<endl;
        //then do fprop
        fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);
        //then compute NCE loss function
        perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                        hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                        unigram_probs_vector,D_prime);
        finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
        difference = finite_difference_gradient - gradient_hidden_to_output_matrix(row_perturb_dimension,col_perturb_dimension);
        cout<<"the ratio is "<<gradient_hidden_to_output_matrix(row_perturb_dimension,col_perturb_dimension)/finite_difference_gradient<<endl;
        cout<<"the difference for hidden to output context matrix was "<<abs(difference)<<endl;
        hidden_layer_to_output_node.param->U(row_perturb_dimension,col_perturb_dimension) -= delta_perturb;
        getchar();

        //restoring the fprop to the original one
        initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
        fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);
        int example_id = rand() % current_minibatch_size;
        //now pick that example , perturb, do fprop and check the gradient
        Matrix<double,1,Dynamic> perturb_vector(embedding_dimension);
        int perturb_dimension = rand() % embedding_dimension;
        cout<<"the perturb dimension was "<<perturb_dimension<<endl;
        int output_word = shuffled_training_data[myParam.ngram_size-1](minibatch_start_index+example_id);
        cout<<"the output word was "<<output_word<<endl;
        D_prime.W(output_word,perturb_dimension) += delta_perturb;
        perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                            hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                            unigram_probs_vector,D_prime);
        //cout<<"the perturbed nce loss was "<<perturbed_nce_loss<<endl;
        //cout<<"the finite difference gradient of the perturb dimension "<<perturb_dimension<<" was "<<(perturbed_nce_loss-current_nce_loss)/delta_perturb<<endl;
        finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
        difference = finite_difference_gradient - gradient_output_W(output_word,perturb_dimension);
        cout<<"the ratio is "<<gradient_output_W(output_word,perturb_dimension)/finite_difference_gradient<<endl;
        cout<<"the difference for output W was "<<abs(difference)<<endl;
        D_prime.W(output_word,perturb_dimension) -= delta_perturb;
        getchar();


        //now checking the gradient for output bias
        D_prime.b(output_word) += delta_perturb;
        perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                            hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                            unigram_probs_vector,D_prime);
        finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
        difference = finite_difference_gradient - gradient_output_b(output_word);
        cout<<"the ratio is "<<gradient_output_b(output_word)/finite_difference_gradient<<endl;
        cout<<"the difference for output b was "<<abs(difference)<<endl;
        D_prime.b(output_word) -= delta_perturb;
        getchar();
  
        //checking the gradient for one of the words in the noise samples
        int noise_word_id = rand()%num_noise_samples;
        cout<<"the noise word id was "<<noise_word_id<<endl;
        int noise_word = minibatch_negative_samples(example_id,noise_word_id);
        cout<<"the noise word was "<<noise_word<<endl;
        D_prime.W(noise_word,perturb_dimension) += delta_perturb;
        perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                            hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                            unigram_probs_vector,D_prime);
        finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
        difference = finite_difference_gradient - gradient_output_W(noise_word,perturb_dimension);
        cout<<"the ratio is "<<gradient_output_W(noise_word,perturb_dimension)/finite_difference_gradient<<endl;
        cout<<"the difference for output noise W was "<<abs(difference)<<endl;
        D_prime.W(noise_word,perturb_dimension) -= delta_perturb;
        getchar();

        //now checking the gradient for hbias
        //cout<<"gradient h bias is "<<gradient_h_bias<<endl;
        initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
        perturb_dimension = rand() % n_hidden;
        //cout<<"the perturb dimension is "<<perturb_dimension<<endl;
        //out<<"h bias before perturbing is "<<hidden_layer_node.param->h_bias<<endl;
        hidden_layer_node.param->h_bias(perturb_dimension) += delta_perturb;
        //cout<<"h bias after perturbing is "<<hidden_layer_node.param->h_bias<<endl;
        fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);
        perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                                hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                                unigram_probs_vector,D_prime);
        cout<<"the perturbed loss was "<<perturbed_nce_loss<<endl;
        finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/ delta_perturb;
        cout<<"the finited difference gradient for h bias was "<<finite_difference_gradient<<endl;
        difference = finite_difference_gradient -gradient_h_bias(perturb_dimension);
        cout<<"the ratio is "<<gradient_h_bias(perturb_dimension)/finite_difference_gradient<<endl;
        cout<<"the difference for h_bias was was "<<abs(difference)<<endl;
        hidden_layer_node.param->h_bias(perturb_dimension) -=  delta_perturb;
        getchar();
        for (int word = 0;word<myParam.ngram_size-1;word++)
        {
            cout<<"the word is "<<word<<endl;
            for (int num_perturb = 0;num_perturb<3;num_perturb++)
            {
                initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
                perturb_dimension = rand() % embedding_dimension;
                cout<<"the perturb dimension was "<<perturb_dimension<<endl;
                //first perturb
                int input_word = shuffled_training_data[word](minibatch_start_index+example_id);
                //cout<<"the input word was "<<input_word<<endl;
                //cout<<"before perturbation the dimension was "<< word_nodes[word].param->W(input_word,perturb_dimension)<<endl;
                word_nodes[word].param->W(input_word,perturb_dimension) += delta_perturb;
                //cout<<"after perturbation the dimension was "<< word_nodes[word].param->W(input_word,perturb_dimension)<<endl;
                //then do fprop
                fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);
                //then compute NCE loss function
                double perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                                hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                                unigram_probs_vector,D_prime);

                //cout<<"the perturbed nce loss was "<<perturbed_nce_loss<<endl;
                finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
                difference = finite_difference_gradient - gradient_input_W(input_word,perturb_dimension);
                cout<<"the ratio is "<<gradient_input_W(input_word,perturb_dimension)/finite_difference_gradient<<endl;
                cout<<"the difference for input W was was "<<abs(difference)<<endl;

                //cout<<"the finite difference gradient of the perturb dimension "<<perturb_dimension<<" was "<<(perturbed_nce_loss-current_nce_loss)/delta_perturb<<endl;
                //cout<<"the gradient was "<<gradient_input_W(input_word,perturb_dimension);
                //unpurturbing
                word_nodes[word].param->W(input_word,perturb_dimension)-= delta_perturb;
                if (abs(difference) > 10E-6)
                {
                    cout<<"the difference was greater than 10E-6 and the original paramter was "<<word_nodes[word].param->W(input_word,perturb_dimension)<<endl;
                }

                getchar();
            }
            //now perturbing the U matrices and checking gradients via finite differences
            for (int num_perturb = 0;num_perturb<3;num_perturb++)
            {
                initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);
                int row_perturb_dimension = rand() % n_hidden;
                cout<<"the row perturb dimension was "<<row_perturb_dimension<<endl;
                int col_perturb_dimension = rand() % embedding_dimension;
                cout<<"the col perturb dimension was "<<col_perturb_dimension<<endl;

                //first perturb
                //cout<<"before perturbation the dimension was "<< context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension)<<endl;
                context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension) += delta_perturb;
                //cout<<"after perturbation the dimension was "<< context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension)<<endl;
                //then do fprop
                fPropGradCheck(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,shuffled_training_data);
                //then compute NCE loss function
                double perturbed_nce_loss = computeLossFunction(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,
                                hidden_layer_to_output_node,shuffled_training_data,c_h,hidden_layer_to_output_node.fProp_matrix,minibatch_negative_samples,
                                unigram_probs_vector,D_prime);
                finite_difference_gradient = (perturbed_nce_loss-current_nce_loss)/delta_perturb;
                difference = finite_difference_gradient - gradients_context_matrix[word](row_perturb_dimension,col_perturb_dimension);
                cout<<"the ratio is "<<gradients_context_matrix[word](row_perturb_dimension,col_perturb_dimension)/finite_difference_gradient<<endl;
                cout<<"the difference for context matrix was "<<abs(difference)<<endl;
                //cout<<"the perturbed nce loss was "<<perturbed_nce_loss<<endl;
                //cout<<"the finite difference gradient of the perturb dimension "<<perturb_dimension<<" was "<<(perturbed_nce_loss-current_nce_loss)/delta_perturb<<endl;
                //cout<<"the gradient was "<<gradients_context_matrix[word](row_perturb_dimension,col_perturb_dimension);
                //unpurturbing
                context_nodes[word].param->U(row_perturb_dimension,col_perturb_dimension) -= delta_perturb;


                getchar();
            }
            
        }
        
    }
    
    initZero(myParam,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node);

}

void initZero(param & myParam,vector<word_node > &word_nodes,vector<context_node > &context_nodes,hidden_node &hidden_layer_node,
              context_node & hidden_layer_to_output_node)
{
    //for all the nodes in the graph, I have to set the bprop and fprop matrices to zero
    hidden_layer_node.bProp_matrix.setZero();
    hidden_layer_node.fProp_matrix.setZero();
    hidden_layer_to_output_node.fProp_matrix.setZero();
    hidden_layer_to_output_node.bProp_matrix.setZero();

    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        context_nodes[word].fProp_matrix.setZero();
        context_nodes[word].bProp_matrix.setZero();
        word_nodes[word].fProp_matrix.setZero();
        //word_nodes[word].bProp_matrix.setZero();
    }
    
}
double computeLossFunction(param & myParam,int minibatch_start_index,int current_minibatch_size,vector<word_node > &word_nodes,
            vector<context_node > &context_nodes,hidden_node &hidden_layer_node,context_node & hidden_layer_to_output_node,
            vector<Matrix<int,Dynamic,1> >&shuffled_training_data,vector_map & c_h,Matrix<double,Dynamic,Dynamic> & minibatch_predicted_embeddings,
            Matrix<int,Dynamic,Dynamic> & minibatch_negative_samples,vector<vector<double> > &unigram_probs_vector,Output_word_embeddings & D_prime)
{
    std::setprecision(9);
    int ngram_size = myParam.ngram_size;
    int n_vocab = myParam.n_vocab;
    int embedding_dimension = myParam.embedding_dimension;
    int minibatch_size = myParam.minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;
    double normalization_init = myParam.normalization_init;

    //parallelizing the creation with multithreading
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    double minibatch_loss;
    #pragma omp parallel firstprivate(minibatch_size,minibatch_start_index,ngram_size,n_vocab,embedding_dimension, \
                            num_noise_samples,normalization_init)
    {
    #pragma omp for reduction(+:minibatch_loss) //schedule(dynamic)
    for (int train_id = 0;train_id < minibatch_size;train_id++)
    {

        int thread_id = omp_get_thread_num();
        int output_word = shuffled_training_data[ngram_size-1](minibatch_start_index+train_id);
        //cout<<"output word is "<<output_word<<endl;
        Matrix<double,Dynamic,1> predicted_embedding = minibatch_predicted_embeddings.col(train_id);
        //cout<<"predicted embedding is "<<endl<<predicted_embedding<<endl;

        vector<int> context;//(ngram_size-1);

        //creating the context
        for (int word = 0;word<ngram_size-1;word++)
        {
            context.push_back(shuffled_training_data[word](minibatch_start_index+train_id));
            //context.push_back((*thread_data_col_locations[thread_id][word])(minibatch_start_index+train_id));
        }

        double log_inv_normalization_const_h = 0.;
        //getting a normalization constant and making it threasafe
        
        //this region does not need to be critical because its just a read
        log_inv_normalization_const_h = c_h[context];

        double inv_normalization_const_h = exp(log_inv_normalization_const_h);
        //cout<<"The normalization constant is "<<inv_normalization_const_h<<endl;
        //setting the gradient for that context to 0;
        //double c_h_gradient_vector = 0.0;

        double score = D_prime.W.row(output_word).dot(predicted_embedding) + D_prime.b(output_word);
        //cout<<"the positive score is "<<score<<endl;
        double unnorm_positive_prob = exp(score);
        //cout<<"the unnorm positive prob is "<<unnorm_positive_prob<<endl;
        //cout<<"the unigram prob is "<<unigram_probs_vector[thread_id][output_word]<<endl;
        //cout<<"the positive prob is "<<
        double sample_loss = 0.;
        double positive_prob = (unnorm_positive_prob*inv_normalization_const_h/
                        (unnorm_positive_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][output_word]));
        //cout<<"positive prob is "<<positive_prob<<endl;
        sample_loss = log(positive_prob);
        //cout<<"sample loss is "<<sample_loss<<endl;

       
        ///COMPUTING NOISE SAMPLES///
        for (int sample_id = 0;sample_id <num_noise_samples;sample_id++)
        {

            int sample = minibatch_negative_samples(train_id,sample_id);
            assert (sample >= 0);
            double negative_score = D_prime.W.row(sample).dot(predicted_embedding) + D_prime.b(sample);
            //cout<<"the negative score is "<<negative_score<<endl;
            //cout<<"the sample was "<<sample<<endl;
            double negative_unnorm_prob = exp(negative_score);
            double negative_prob = num_noise_samples*unigram_probs_vector[thread_id][sample]/
                              (negative_unnorm_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][sample]);
            //cout<<"negative prob is "<<negative_prob<<endl;
            sample_loss += log(negative_prob);
        }
        //cout<<"the sample loss is "<<sample_loss<<endl;

        minibatch_loss += sample_loss; 
    }
    }
    #pragma omp barrier
    return(minibatch_loss);
}

void inline fPropGradCheck(param & myParam,int minibatch_start_index,int current_minibatch_size,vector<word_node > &word_nodes,
            vector<context_node > &context_nodes,hidden_node &hidden_layer_node,context_node & hidden_layer_to_output_node,
            vector<Matrix<int,Dynamic,1> >&data)
{
    /////FORWARD PROPAGATION/////////////
    //doing forward propagation first with word nodes
    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        word_nodes[word].param->fPropOmp(data[word],word_nodes[word].fProp_matrix,minibatch_start_index,current_minibatch_size);
    }

    Eigen::setNbThreads(myParam.n_threads);
    //doing forward prop with the context nodes
    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        context_nodes[word].param->fProp(word_nodes[word].fProp_matrix,context_nodes[word].fProp_matrix);
        //cout<<"context fprop matrix was "<<context_nodes[word].fProp_matrix<<endl;

    }
    //doing forward prop with the hidden nodes
    hidden_layer_node.param->fPropTanh(context_nodes,hidden_layer_node.fProp_matrix);
    /*
    Matrix<double,Dynamic,Dynamic> hidden_layer_input(myParam.n_hidden,myParam.embedding_dimension);
    for (int word = 0;word<myParam.ngram_size-1;word++)
    {
        hidden_layer_node.param->fPropTanh(context_nodes[word].fProp_matrix,hidden_layer_node.fProp_matrix);
    }
    */
    //cout<<"the hidden fprop matrix was "<<hidden_layer_node.fProp_matrix<<endl;

    //now doing forward prop with the hidden to output nodes

    hidden_layer_to_output_node.param->fProp(hidden_layer_node.fProp_matrix,hidden_layer_to_output_node.fProp_matrix);

    //cout<<"the hidden to output fprop matrix was "<<hidden_layer_to_output_node.fProp_matrix<<endl;

}

