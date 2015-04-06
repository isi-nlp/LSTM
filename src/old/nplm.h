#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
//#include <random>
#include "param.h"
#include <ctime>
#include <stdio.h>
//#include <chrono>
#include <math.h>
#include "util.h"
#include <iomanip>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "maybe_omp.h"
#include <algorithm>
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <assert.h>
using namespace std;
using namespace Eigen;
using namespace boost::random;

typedef double Real;
typedef Matrix<bool,1 , Dynamic> vectorHidden;

class RBM
{
    
    private:
        vectorHidden hidden_layer;
        Matrix<double,Dynamic,Dynamic> W;
        Matrix<double,Dynamic,Dynamic> U;
        Matrix<double,Dynamic,Dynamic> velocity_U;
        Matrix<double,1,Dynamic> v_bias;
        Matrix<double,1,Dynamic> h_bias;
        //in order to make this parallel, I will create a vector of parameters 
        //equal to the number of threads
        vector<Matrix<double,Dynamic,Dynamic> > W_vector;
        vector<Matrix<double,Dynamic,Dynamic> > U_vector;
        vector<Matrix<double,1,Dynamic> > v_bias_vector;
        vector<Matrix<double,1,Dynamic> > h_bias_vector;

    public:
        //initializing directly
        RBM(Matrix<double,Dynamic,Dynamic> input_W,Matrix<double,Dynamic,Dynamic> input_U,
            Matrix<double,1,Dynamic> input_v_bias, Matrix<double,1,Dynamic> input_h_bias,int n_threads)
        {
            W = input_W;
            U = input_U;
            v_bias = input_v_bias;
            h_bias = input_h_bias;

            //unsigned int num_threads = omp_get_num_threads();
            //cerr<<"num threads is "<<num_threads<<endl;
            for (int i = 0;i<n_threads;i++)
            {
                cerr<<"i is "<<i<<endl;
                Matrix<double,Dynamic,Dynamic> temp_W = W;
                W_vector.push_back(temp_W);
                //cerr<<"W vec element "<<i<<" is "<<W_vector[i]<<endl;
                Matrix<double,Dynamic,Dynamic> temp_U = U;
                U_vector.push_back(temp_U);
                //cerr<<"U vec element "<<i<<" is "<<U_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_v_bias = v_bias;
                v_bias_vector.push_back(temp_v_bias);
                //cerr<<"v bias element "<<i<<" is "<<v_bias_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_h_bias = h_bias;
                h_bias_vector.push_back(temp_h_bias);
                //cerr<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
            }

        }
        //RBM(param myParam);
        RBM(param myParam)
        {
            cerr<<"in the RBM constructor"<<endl;
            //initializing the weights et
            W.resize(myParam.n_vocab,myParam.embedding_dimension);
            U.resize(myParam.n_hidden,myParam.embedding_dimension*myParam.ngram_size);
            velocity_U.setZero(myParam.n_hidden,myParam.embedding_dimension*myParam.ngram_size);
            v_bias.resize(1,myParam.n_vocab);
            h_bias.resize(1,myParam.n_hidden);
            /*
            //reading the params from the given files
            readParameter("random_nos.W", W);
            cerr<<"read W"<<endl;
            //cerr<<W<<endl;
            readParameter("random_nos.U", U);
            cerr<<"read U"<<endl;
            //cerr<<U<<endl;
            readParameterBias("random_nos.v_bias", v_bias);
            cerr<<"read v_bias"<<endl;
            //cerr<<v_bias<<endl;
            readParameterBias("random_nos.h_bias", h_bias);
            cerr<<"read h_bias"<<endl;
            //cerr<<h_bias<<endl;
            //getchar();
            //we have to initialize W and U 
            //unsigned seed = chrono::system_clock::now().time_since_epoch().count();
            */
			      unsigned seed = std::time(0);
            clock_t t;
            /*
            //unsigned seed = 1234; //for testing i have a constant seed
            mt19937 eng_W (seed);  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_U (seed);  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_h (seed);  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_v (seed);  // mt19937 is a standard mersenne_twister_engine
            */
            cerr<<t+rand()<<endl;
            cerr<<t+rand()<<endl;
            mt19937 eng_W (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_U (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_h (t+rand());  // mt19937 is a standard mersenne_twister_engine
            mt19937 eng_bias_v (t+rand());  // mt19937 is a standard mersenne_twister_engine
            cerr<<"W rows is "<<W.rows()<<" and W cols is "<<W.cols()<<endl;

            cerr<<"U rows is "<<U.rows()<<" and U cols is "<<U.cols()<<endl;

            cerr<<"v_bias rows is "<<v_bias.rows()<<" and v_bias cols is "<<v_bias.cols()<<endl;

            cerr<<"h_bias rows is "<<h_bias.rows()<<" and h_bias cols is "<<h_bias.cols()<<endl;

            void * distribution ;
            if (myParam.init_normal == 0)
            {
                uniform_real_distribution<> unif_real(-0.01, 0.01); 
                //initializing W
                for (int i =0;i<W.rows();i++)
                {
                    //cerr<<"i is "<<i<<endl;
                    for (int j =0;j<W.cols();j++)
                    {
                        //cerr<<"j is "<<j<<endl;
                        W(i,j) = unif_real(eng_W);    
                    }
                }
                //initializing U

                for (int i =0;i<U.rows();i++)
                {
                    for (int j =0;j<U.cols();j++)
                    {
                        U(i,j) = unif_real(eng_U);    
                    }
                }
                //initializing v_bias
                for (int i =0;i<v_bias.cols();i++)
                {
                    //cerr<<"i is "<<i<<endl;
                    v_bias(i) = unif_real(eng_bias_v);
                }

                //initializing h_bias
                for (int i =0;i<h_bias.cols();i++)
                {
                    h_bias(i) = unif_real(eng_bias_h);
                }

            }
            else //initialize with gaussian distribution with mean 0 and stdev 0.01
            {
                normal_distribution<double> unif_normal(0.,0.01);
                //initializing W
                for (int i =0;i<W.rows();i++)
                {
                    //cerr<<"i is "<<i<<endl;
                    for (int j =0;j<W.cols();j++)
                    {
                        //cerr<<"j is "<<j<<endl;
                        W(i,j) = unif_normal(eng_W);    
                    }
                }
                //initializing U

                for (int i =0;i<U.rows();i++)
                {
                    for (int j =0;j<U.cols();j++)
                    {
                        U(i,j) = unif_normal(eng_U);    
                    }
                }
                //initializing v_bias
                for (int i =0;i<v_bias.cols();i++)
                {
                    //cerr<<"i is "<<i<<endl;
                    v_bias(i) = unif_normal(eng_bias_v);
                }

                //initializing h_bias
                for (int i =0;i<h_bias.cols();i++)
                {
                    h_bias(i) = unif_normal(eng_bias_h);
                }
              
            }
            //unsigned int num_threads = omp_get_num_threads();
            //cerr<<"num threads is "<<num_threads<<endl;
            for (int i = 0;i<myParam.n_threads;i++)
            {
                cerr<<"i is "<<i<<endl;
                Matrix<double,Dynamic,Dynamic> temp_W = W;
                W_vector.push_back(temp_W);
                //cerr<<"W vec element "<<i<<" is "<<W_vector[i]<<endl;
                Matrix<double,Dynamic,Dynamic> temp_U = U;
                U_vector.push_back(temp_U);
                //cerr<<"U vec element "<<i<<" is "<<U_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_v_bias = v_bias;
                v_bias_vector.push_back(temp_v_bias);
                //cerr<<"v bias element "<<i<<" is "<<v_bias_vector[i]<<endl;
                Matrix<double,1,Dynamic> temp_h_bias = h_bias;
                h_bias_vector.push_back(temp_h_bias);
                //cerr<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
            }
            /*
            cerr<<"the element is "<<W_vector[0].row(1)<<endl;
            W_vector[0](1,0) = 1.5;
            cerr<<"the W element after is "<<W.row(1)<<endl;
            cerr<<"the 0 element after is "<<W_vector[0].row(1)<<endl;
            cerr<<"the 1 element after is "<<W_vector[1].row(1)<<endl;
            getchar();
            */
        }
        
        void sample_h_given_v_omp(Matrix<int,Dynamic,Dynamic>& v_layer_minibatch,int minibatch_size,int minibatch_start_index,
            Matrix<bool,Dynamic,Dynamic>& h_layer_minibatch,Matrix<double,Dynamic,Dynamic> &h_layer_probs_minibatch, param & myParam,
            vector<uniform_real_distribution<> >& unif_real_vector,vector<mt19937 >& eng_real_vector,int current_cdk,
            vector<double>  &random_nos_real,vector<int> &random_nos_int,int *random_nos_real_counter,int* random_nos_int_counter)
        {

            //cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
            //cerr<<"U dimension is "<<U.rows()<<" and "<<U.cols()<<endl;
            //cerr<<"the size of h logit is "<<h_logit.rows()<<" and "<<h_logit.cols()<<endl;
            //cerr<<"the number of threads is "<<omp_get_num_threads();
            //getchar();
            Eigen::initParallel();
            #pragma omp parallel  shared(h_layer_minibatch,h_layer_probs_minibatch,myParam,v_layer_minibatch,unif_real_vector,eng_real_vector) \
                                firstprivate(minibatch_start_index,minibatch_size)
            {
                /*
                clock_t t;
                t = clock();
                mt19937 eng_real_temp (t);  // mt19937 is a standard mersenne_twister_engine
                uniform_real_distribution<> unif_real_temp(0.0, 1.0);
                int thread_number = omp_get_thread_num();
                */
            int embedding_dimension = myParam.embedding_dimension;
            int ngram_size = myParam.ngram_size;
            int n_hidden = myParam.n_hidden;
           
            #pragma omp for            

            for (int i = 0;i< minibatch_size;i++)
            {
                /*
                #pragma omp master
                {
                    cerr<<"master start"<<endl;
                    cerr<<"the number of threads is "<<omp_get_num_threads()<<endl;
                    cerr<<"i is "<<i<<endl;
                    cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
                    cerr<<"master end"<<endl;
                }
                */
                int thread_id = omp_get_thread_num();
                /*
                if (thread_id == 1)
                {
                    cerr<<"the minibatch index for that is "<<minibatch_start_index<<endl;
                    cerr<<"the i index for thread 1 is "<<i<<endl;
                }
                */
                Matrix<double,Dynamic,1> h_logit;
                h_logit.setZero(n_hidden);
                /*
                if (thread_id == 1)
                {
                    cerr<<"we just created h logit and i was "<<i<<endl;
                }
                */
                //cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                //cerr<<"h logit is "<<h_logit<<endl;
                //summing up over each word in the training ngram to get the total logit
                /*
                if (thread_id == 1 || thread_id == 0)
                {
                    cerr<<"the current training example index is "<<i+minibatch_start_index<<endl;
                    cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                }
                */

                for (int index = 0;index < ngram_size;index++)
                {
                    //cerr<<"the w row is "<<W.row(v_layer_minibatch(minibatch_start_index + i,index))<<endl;
                    h_logit += U_vector[thread_id].block(0,index*embedding_dimension,n_hidden,embedding_dimension)*W_vector[thread_id].row(v_layer_minibatch(minibatch_start_index + i,index)).transpose();
                    //cerr<<"after performing the sum, h logit is "<<h_logit<<endl;
                }
                h_logit += h_bias_vector[thread_id];
                /*
                if (thread_id == 1)
                {
                    cerr<<"we just finalized h logit and i was "<<i<<endl;
                }
                */
                //cerr<<"before current minibatch h probs was "<<(*current_h_minibatch_probs_pointer).row(i)<<endl;
                h_layer_probs_minibatch.row(i) = (1/(1+(-(h_logit.array())).exp()));
                //cerr<<"current minibatch h probs is "<<(*current_h_minibatch_probs_pointer).row(i)<<endl;
                //cerr<<"h logit was "<<h_logit<<endl;
                //cerr<<"finished assignment "<<endl;
                //cerr<<"curret minibatch h states is "<<(*current_h_minibatch_pointer).row(i)<<endl;
                //cerr<<"after assignment "<<h_layer_probs_minibatch.row(i)<<endl;
                /*
                if (thread_id == 1 || thread_id == 0)
                {
                    cerr<<"the current training example index is "<<i+minibatch_start_index<<endl;
                    cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                    cerr<<"h logit was "<<h_logit<<endl;
                    cerr<<"after assignment, the probs is "<<h_layer_probs_minibatch.row(i)<<endl;

                }
                */
                for (int index = 0;index < n_hidden;index++)
                {
                    //double p = unif_p(eng);
                    //double p = unif_real_temp(eng_real_temp);
                    double p = unif_real_vector[thread_id](eng_real_vector[thread_id]);
                    //double p = random_nos_real[*random_nos_real_counter];
                    //(*random_nos_real_counter) += 1;
                    //cerr<<"value of p was "<<p<<endl;
                    if (h_layer_probs_minibatch(i,index) >= p)
                    {
                        h_layer_minibatch(i,index) = 1;
                    }
                    else
                    {
                        h_layer_minibatch(i,index) = 0;
                    }
                    //cerr<<"done getting a state"<<endl;
                }
                /*
                if (thread_id == 1 || thread_id ==0 )
                {
                    cerr<<"we are done getting the states from thread id "<<i<<endl;
                }
                */
                //cerr<<"we are done getting the states"<<endl;
                //cerr<<"the hidden state was "<<h_layer_minibatch.row(i)<<endl;
                //cerr<<"the number of ones was "<<h_layer_minibatch.cast<double>().row(i).sum()<<endl;
                //getchar();

            }
            }
            #pragma omp barrier
            //cerr<<"we just finished a minibatch"<<endl;
            //cerr<<"the new h states is "<<(*current_h_minibatch_pointer)<<endl; 
        }

        void sample_h_given_v(Matrix<int,Dynamic,Dynamic>& v_layer_minibatch,int minibatch_size,int minibatch_start_index,
            Matrix<bool,Dynamic,Dynamic>& h_layer_minibatch,Matrix<double,Dynamic,Dynamic> &h_layer_probs_minibatch, param & myParam,
            uniform_real_distribution<> & unif_p,mt19937 & eng,int current_cdk,vector<double>  &random_nos_real,vector<int> &random_nos_int,
            int *random_nos_real_counter,int* random_nos_int_counter)
        {

            //cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
            //cerr<<"U dimension is "<<U.rows()<<" and "<<U.cols()<<endl;
            //cerr<<"the size of h logit is "<<h_logit.rows()<<" and "<<h_logit.cols()<<endl;
            //cerr<<"the number of threads is "<<omp_get_num_threads();
            //getchar();
            Eigen::initParallel();
            #pragma omp parallel  shared(h_layer_minibatch,h_layer_probs_minibatch,myParam,v_layer_minibatch,unif_p,eng) \
                                firstprivate(minibatch_start_index,minibatch_size)
            {

                clock_t t;
                t = clock();
                mt19937 eng_real_temp (t);  // mt19937 is a standard mersenne_twister_engine
                uniform_real_distribution<> unif_real_temp(0.0, 1.0);
            int embedding_dimension = myParam.embedding_dimension;
            int ngram_size = myParam.ngram_size;
            int n_hidden = myParam.n_hidden;


            
            #pragma omp for            
            //#pragma master
            //{
            for (int i = 0;i< minibatch_size;i++)
            {

                /*
                #pragma omp master
                {
                    cerr<<"master start"<<endl;
                    cerr<<"the number of threads is "<<omp_get_num_threads()<<endl;
                    cerr<<"i is "<<i<<endl;
                    cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
                    cerr<<"master end"<<endl;
                }
                */
                int thread_id = omp_get_thread_num();
                /*
                if (thread_id == 1)
                {
                    cerr<<"the minibatch index for that is "<<minibatch_start_index<<endl;
                    cerr<<"the i index for thread 1 is "<<i<<endl;
                }
                */
                Matrix<double,Dynamic,1> h_logit;
                h_logit.setZero(n_hidden);
                /*
                if (thread_id == 1)
                {
                    cerr<<"we just created h logit and i was "<<i<<endl;
                }
                */
                //cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                //cerr<<"h logit is "<<h_logit<<endl;
                //summing up over each word in the training ngram to get the total logit
                /*
                if (thread_id == 1 || thread_id == 0)
                {
                    cerr<<"the current training example index is "<<i+minibatch_start_index<<endl;
                    cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                }
                */

                for (int index = 0;index < ngram_size;index++)
                {
                    //cerr<<"the w row is "<<W.row(v_layer_minibatch(minibatch_start_index + i,index))<<endl;
                    h_logit += U.block(0,index*embedding_dimension,n_hidden,embedding_dimension)*W.row(v_layer_minibatch(minibatch_start_index + i,index)).transpose();
                    //cerr<<"after performing the sum, h logit is "<<h_logit<<endl;
                }
                h_logit += h_bias;
                /*
                if (thread_id == 1)
                {
                    cerr<<"we just finalized h logit and i was "<<i<<endl;
                }
                */
                //cerr<<"before current minibatch h probs was "<<(*current_h_minibatch_probs_pointer).row(i)<<endl;
                h_layer_probs_minibatch.row(i) = (1/(1+(-(h_logit.array())).exp()));
                //cerr<<"current minibatch h probs is "<<(*current_h_minibatch_probs_pointer).row(i)<<endl;
                //cerr<<"h logit was "<<h_logit<<endl;
                //cerr<<"finished assignment "<<endl;
                //cerr<<"curret minibatch h states is "<<(*current_h_minibatch_pointer).row(i)<<endl;
                //cerr<<"after assignment "<<h_layer_probs_minibatch.row(i)<<endl;
                /*
                if (thread_id == 1 || thread_id == 0)
                {
                    cerr<<"the current training example index is "<<i+minibatch_start_index<<endl;
                    cerr<<"the current training example is "<<v_layer_minibatch.row(i+minibatch_start_index)<<endl;
                    cerr<<"h logit was "<<h_logit<<endl;
                    cerr<<"after assignment, the probs is "<<h_layer_probs_minibatch.row(i)<<endl;

                }
                */
                for (int index = 0;index < n_hidden;index++)
                {
                    //double p = unif_p(eng);
                    double p = unif_real_temp(eng_real_temp);
                    //double p = random_nos_real[*random_nos_real_counter];
                    //(*random_nos_real_counter) += 1;
                    //cerr<<"value of p was "<<p<<endl;
                    if (h_layer_probs_minibatch(i,index) >= p)
                    {
                        h_layer_minibatch(i,index) = 1;
                    }
                    else
                    {
                        h_layer_minibatch(i,index) = 0;
                    }
                    //cerr<<"done getting a state"<<endl;
                }
                /*
                if (thread_id == 1 || thread_id ==0 )
                {
                    cerr<<"we are done getting the states from thread id "<<i<<endl;
                }
                */
                //cerr<<"we are done getting the states"<<endl;
                //cerr<<"the hidden state was "<<h_layer_minibatch.row(i)<<endl;
                //cerr<<"the number of ones was "<<h_layer_minibatch.cast<double>().row(i).sum()<<endl;
                //getchar();

            }
            }
            //}
            #pragma omp barrier
            //cerr<<"we just finished a minibatch"<<endl;
            //cerr<<"the new h states is "<<(*current_h_minibatch_pointer)<<endl; 
        }



        void sample_v_given_h_omp(Matrix<int,Dynamic,Dynamic> &v_layer_minibatch,Matrix<bool,Dynamic,Dynamic> &h_layer_minibatch,int minibatch_size,
                             param & myParam,vector<uniform_real_distribution<> >& unif_real_vector,vector<mt19937> & eng_real_vector,
                             vector<uniform_int_distribution<> > & unif_int_vector,vector<mt19937> & eng_int_vector,vector<vector<double> > &unigram_probs_vector,
                             vector<vector<double> > & q_vector,vector<vector<int> >&J_vector,int current_cdk,vector<double> &random_nos_real,
                             vector<int> &random_nos_int,int *random_nos_real_counter,int* random_nos_int_counter)
        {

            Eigen::initParallel();
            #pragma omp parallel  shared(h_layer_minibatch,myParam,v_layer_minibatch,unif_real_vector,eng_real_vector,unif_int_vector,eng_int_vector) \
                                firstprivate(minibatch_size,current_cdk)
            {
                /*
                clock_t t;
                t = clock();
                mt19937 eng_real_temp (t);  // mt19937 is a standard mersenne_twister_engine
                uniform_real_distribution<> unif_real_temp(0.0, 1.0);
                mt19937 eng_int_temp (t);  // mt19937 is a standard mersenne_twister_engine
                uniform_int_distribution<> unif_int_temp(0, myParam.n_vocab-1);
                */
                int embedding_dimension = myParam.embedding_dimension;
                int ngram_size = myParam.ngram_size;
                int n_hidden = myParam.n_hidden;
                int mh_steps = myParam.mh_steps;
           
            #pragma omp for               
            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                //carrying out myParam.mh_steps of metropolis hastings
                for (int mh_step = 0;mh_step < mh_steps;mh_step++)
                {
                    //performing mh sampling one index at a time
                    for (int index = 0;index<ngram_size;index++)
                    {
                        //int mixture_component = unif_int_temp(eng_int_temp);

                        int mixture_component = unif_int_vector[thread_id](eng_int_vector[thread_id]);
                        //int mixture_component = random_nos_int[*random_nos_int_counter];
                        //(*random_nos_int_counter) += 1;
                        //cerr<<"the mixture component was "<<mixture_component<<endl;
                        //cerr<<"l component is "<<J[mixture_component]<<endl;
                        //double p = unif_real_temp(eng_real_temp);

                        double p = unif_real_vector[thread_id](eng_real_vector[thread_id]);
                        //double p = random_nos_real[*random_nos_real_counter];
                        //cerr<<"p is "<<p<<endl;
                        //(*random_nos_real_counter)+= 1;
                        int sample ;
                        //cerr<<"bernoulli prob is "<<q_vector[thread_id][mixture_component]<<endl;
                        //cerr<<"remaining bernoulli item is "<<J_vector[thread_id][mixture_component]<<endl;
                        if (q_vector[thread_id][mixture_component] >= p)
                        {
                            //cerr<<"mixture accepted"<<endl;
                            sample = mixture_component;
                        }
                        else
                        {
                            //cerr<<"J accepted "<<endl;
                            sample = J_vector[thread_id][mixture_component];
                        }
						assert (sample >= 0);
                        //cerr<<"the sample was "<<sample<<endl;
                        int current_visible_state = v_layer_minibatch(i,index);
                        double accept_reject = (unigram_probs_vector[thread_id][current_visible_state]/unigram_probs_vector[thread_id][sample]) *
                                        exp(h_layer_minibatch.row(i).cast<double>()*
                                        (U_vector[thread_id].block(0,index*embedding_dimension,n_hidden,embedding_dimension)*
                                        (W_vector[thread_id].row(sample)-W_vector[thread_id].row(current_visible_state)).transpose())+
                                        v_bias_vector[thread_id](sample)-v_bias_vector[thread_id](current_visible_state));
                        //cerr<<"regular accept reject is "<<accept_reject<<endl;
                        //cerr<<"pointer accept reject is "<<accept_reject<<endl;
                        //cerr<<"U block is "<<U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension)<<endl;
                        //cerr<<"w row sample is "<<W.row(sample)<<endl;
                        //cerr<<"w row current visible state is "<<W.row(current_visible_state)<<endl;
                        //cerr<<"quantity inside numerator is "<<(*current_h_pointer).row(i).cast<double>()*(U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension)*
                        //                (W.row(sample)-W.row(current_visible_state)).transpose())+v_bias(sample)-v_bias(current_visible_state)<<endl;
                        //cerr<<" non pointer accept reject is "<<accept_reject_non_pointer<<endl;
                        //now to accept or reject it
                        //p = unif_real_temp(eng_real_temp);

                        p = unif_real_vector[thread_id](eng_real_vector[thread_id]);
                        //p =  random_nos_real[*random_nos_real_counter];
                        //(*random_nos_real_counter)+= 1;

                        if (accept_reject >= p)
                        {
                            //minibatch_v_negative(i,index) = sample;
                            v_layer_minibatch(i,index) = sample;
                            //cerr<<"accepted "<<endl;
                        }
                        /*
                        else
                        {
                            cerr<<"rejected"<<endl;
                        }
                        */

                    }  
                }

                //cerr<<"after mh,the current visible vector is "<<v_layer_minibatch.row(i)<<endl;
                //getchar();
                    
            }
            }
            #pragma omp barrier
            
        }

        void sample_v_given_h(Matrix<int,Dynamic,Dynamic> &v_layer_minibatch,Matrix<bool,Dynamic,Dynamic> &h_layer_minibatch,int minibatch_size,
                             param & myParam,uniform_real_distribution<> & unif_p,mt19937 & eng,uniform_int_distribution<> & unif_int,
                             mt19937 & eng_int,vector<double> &unigram_probs,vector<double> & q,vector<int> &J,int current_cdk,
                             vector<double> &random_nos_real,vector<int> &random_nos_int,int *random_nos_real_counter,int* random_nos_int_counter)
        {

            Eigen::initParallel();
            #pragma master
            {
            for (int i = 0;i< minibatch_size;i++)
            {
                //carrying out myParam.mh_steps of metropolis hastings
                for (int mh_step = 0;mh_step < myParam.mh_steps;mh_step++)
                {
                    //performing mh sampling one index at a time
                    for (int index = 0;index<myParam.ngram_size;index++)
                    {
                        int mixture_component = unif_int(eng_int);
                        //int mixture_component = random_nos_int[*random_nos_int_counter];
                        //(*random_nos_int_counter) += 1;
                        //cerr<<"the mixture component was "<<mixture_component<<endl;
                        //cerr<<"l component is "<<J[mixture_component]<<endl;
                        double p = unif_p(eng);
                        //double p = random_nos_real[*random_nos_real_counter];
                        //cerr<<"p is "<<p<<endl;
                        //(*random_nos_real_counter)+= 1;
                        int sample ;
                        //cerr<<"bernoulli prob is "<<q[mixture_component]<<endl;
                        if (q[mixture_component] >= p)
                        {
                            //cerr<<"mixture accepted"<<endl;
                            sample = mixture_component;
                        }
                        else
                        {
                            //cerr<<"J accepted "<<endl;
                            sample = J[mixture_component];
                        }
                        //cerr<<"the sample was "<<sample<<endl;
                        int current_visible_state = v_layer_minibatch(i,index);
                        double accept_reject = (unigram_probs[current_visible_state]/unigram_probs[sample]) *
                                        exp(h_layer_minibatch.row(i).cast<double>()*
                                        (U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension)*
                                        (W.row(sample)-W.row(current_visible_state)).transpose())+
                                        v_bias(sample)-v_bias(current_visible_state));
                        //cerr<<"regular accept reject is "<<accept_reject<<endl;
                        //cerr<<"pointer accept reject is "<<accept_reject<<endl;
                        //cerr<<"U block is "<<U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension)<<endl;
                        //cerr<<"w row sample is "<<W.row(sample)<<endl;
                        //cerr<<"w row current visible state is "<<W.row(current_visible_state)<<endl;
                        //cerr<<"quantity inside numerator is "<<(*current_h_pointer).row(i).cast<double>()*(U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension)*
                        //                (W.row(sample)-W.row(current_visible_state)).transpose())+v_bias(sample)-v_bias(current_visible_state)<<endl;
                        //cerr<<" non pointer accept reject is "<<accept_reject_non_pointer<<endl;
                        //now to accept or reject it
                        p = unif_p(eng);
                        //p =  random_nos_real[*random_nos_real_counter];
                        //(*random_nos_real_counter)+= 1;

                        if (accept_reject >= p)
                        {
                            //minibatch_v_negative(i,index) = sample;
                            v_layer_minibatch(i,index) = sample;
                            //cerr<<"accepted "<<endl;
                        }
                        /*
                        else
                        {
                            cerr<<"rejected"<<endl;
                        }
                        */
                    }  
                }

                //cerr<<"after mh,the current visible vector is "<<v_layer_minibatch.row(i)<<endl;
                //getchar();
                    
            }
            }
            
        }

        //update the where the user supplies the negative and positive minibatches
        void updateParameters_omp(Matrix<int,Dynamic,Dynamic> &positive_v_minibatch,int minibatch_size,int minibatch_start_index,
                            Matrix<int,Dynamic,Dynamic> &negative_v_minibatch, Matrix<double,Dynamic,Dynamic> &positive_h_probs_minibatch,
                            Matrix<double,Dynamic,Dynamic> &negative_h_probs_minibatch,param & myParam,double current_momentum)
        {
            Matrix<double,Dynamic,Dynamic> U_grad;
            U_grad.setZero(U.rows(),U.cols());
            bool use_momentum = myParam.use_momentum;
            int embedding_dimension = myParam.embedding_dimension;
            int ngram_size = myParam.ngram_size;
            int n_hidden = myParam.n_hidden;
            double L2_reg = myParam.L2_reg;
            double adjusted_learning_rate = myParam.learning_rate/minibatch_size;
            int u_update_threads = (myParam.n_threads > 3)? 3: myParam.n_threads;
            //cerr<<"number of u update threads is "<<u_update_threads<<endl;
            //first get the accumulated gradient for u
            //then update the velocities and the final parameters
            omp_set_num_threads(u_update_threads);
            for (int i = 0;i< minibatch_size;i++)
            {
                Eigen::initParallel();
                #pragma omp parallel  shared(positive_v_minibatch,negative_v_minibatch,positive_h_probs_minibatch,myParam) \
                                firstprivate(minibatch_start_index,minibatch_size,current_momentum,use_momentum,embedding_dimension, \
                                            ngram_size,n_hidden,L2_reg,adjusted_learning_rate)
                {
                #pragma omp for
                for (int index = 0;index < ngram_size;index++)
                {
                    int thread_id = omp_get_thread_num();
                    if (use_momentum == 1)
                    {
                        
                        for (int col = 0;col<embedding_dimension;col++)
                        {
                            U_grad.col(index*embedding_dimension + col) += adjusted_learning_rate*
                                                  (W_vector[thread_id](positive_v_minibatch(i+minibatch_start_index,index),col)*
                                                  positive_h_probs_minibatch.row(i).transpose()-W_vector[thread_id](negative_v_minibatch(i,index),col)*
                                                  negative_h_probs_minibatch.row(i).transpose() - 2*L2_reg*U_vector[thread_id].col(index*embedding_dimension + col));
                        }
                        
                        
                    }
                    else
                    {
                        //cerr<<"no momentum"<<endl;
                        for (int col = 0;col<embedding_dimension;col++)
                        {
                            U.col(index*embedding_dimension + col) += adjusted_learning_rate*
                                                  (W_vector[thread_id](positive_v_minibatch(i+minibatch_start_index,index),col)*
                                                  positive_h_probs_minibatch.row(i).transpose()-W_vector[thread_id](negative_v_minibatch(i,index),col)*
                                                  negative_h_probs_minibatch.row(i).transpose() -2*L2_reg*U_vector[thread_id].col(index*embedding_dimension + col));
                        }
                    }
                    //#pragma omp barrier
                }
                }
            }

            //first we get the u gradients and then we update the velocity     
            if (use_momentum == 1)
            {
                for (int index = 0;index < ngram_size;index++)
                {
                        
                    for (int col = 0;col<embedding_dimension;col++)
                    {

                        velocity_U.col(index*embedding_dimension + col) = current_momentum* velocity_U.col(index*embedding_dimension + col) +
                                                                          U_grad.col(index*embedding_dimension + col);
                        U.col(index*embedding_dimension + col) +=  velocity_U.col(index*embedding_dimension + col);
                    }
                }        
                        
            }

            for (int i = 0;i<myParam.n_threads;i++)
            {
                  U_vector[i] = U;
            }
            Eigen::initParallel();
            omp_set_num_threads(myParam.n_threads);
            //updating the w's
            #pragma omp parallel  shared(positive_v_minibatch,negative_v_minibatch,positive_h_probs_minibatch,myParam) \
                                firstprivate(minibatch_start_index,minibatch_size,current_momentum,use_momentum,embedding_dimension, \
                                            ngram_size,n_hidden,L2_reg,adjusted_learning_rate)
            {
            #pragma omp for
            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                //summing up over each word in the training ngram to get the total logit
                for (int index = 0;index < ngram_size;index++)
                {
                    W.row(positive_v_minibatch(i+minibatch_start_index,index)) += adjusted_learning_rate * 
                                                                        (positive_h_probs_minibatch.row(i)*
                                                                        U_vector[thread_id].block(0,index*embedding_dimension,n_hidden,embedding_dimension));
                    W.row(negative_v_minibatch(i,index)) -= adjusted_learning_rate * 
                                                                        (negative_h_probs_minibatch.row(i)*
                                                                        U_vector[thread_id].block(0,index*embedding_dimension,n_hidden,embedding_dimension));
                }
            }
            #pragma omp barrier

            #pragma omp for
            //updating the v biases
            for (int i = 0;i< minibatch_size;i++)
            {
                int thread_id = omp_get_thread_num();
                //summing up over each word in the training ngram to get the total logit
                for (int index = 0;index < ngram_size;index++)
                {
                    v_bias(positive_v_minibatch(i+minibatch_start_index,index)) += adjusted_learning_rate;
                    v_bias(negative_v_minibatch(i,index)) -= adjusted_learning_rate;

                }
                
            }
            #pragma omp barrier

            #pragma omp master
            {
            for (int i = 0;i< minibatch_size;i++)
            {
                //summing up over each word in the training ngram to get the total logit
                for (int index = 0;index < ngram_size;index++)
                {
                    for (int thread_id = 0;thread_id<myParam.n_threads;thread_id++)
                    {

                        //copying the updated W and v into the thread parameters. This is cheaper since we're only updating the required ones                    
                        W_vector[thread_id].row(positive_v_minibatch(i+minibatch_start_index,index)) = W.row(positive_v_minibatch(i+minibatch_start_index,index));
                        W_vector[thread_id].row(negative_v_minibatch(i,index)) = W.row(negative_v_minibatch(i,index));
                        v_bias_vector[thread_id](positive_v_minibatch(i+minibatch_start_index,index)) = v_bias(positive_v_minibatch(i+minibatch_start_index,index));
                        v_bias_vector[thread_id](negative_v_minibatch(i,index)) = v_bias(negative_v_minibatch(i,index));  
                    }

                }
            }
            }

            }
            //updating the h bias

            for (int i = 0;i< minibatch_size;i++)
            {
                h_bias += adjusted_learning_rate*(positive_h_probs_minibatch.row(i)-negative_h_probs_minibatch.row(i)) ;
            }

            //copying the updated h parameters into the vectors
            for (int thread_id = 0;thread_id<myParam.n_threads;thread_id++)
            {
                  h_bias_vector[thread_id] = h_bias;
            }

        }

        //update the where the user supplies the negative and positive minibatches
        void updateParameters(Matrix<int,Dynamic,Dynamic> &positive_v_minibatch,int minibatch_size,int minibatch_start_index,
                            Matrix<int,Dynamic,Dynamic> &negative_v_minibatch, Matrix<double,Dynamic,Dynamic> &positive_h_probs_minibatch,
                            Matrix<double,Dynamic,Dynamic> &negative_h_probs_minibatch,param & myParam,double current_momentum)
        {
            Matrix<double,Dynamic,Dynamic> U_grad;
            U_grad.setZero(U.rows(),U.cols());

            //my update parameters is incorrect. First I need to accumulate gradients and then I need to add them up 
            //Eigen::initParallel();
            #pragma omp master 
            //cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
            {
            double adjusted_learning_rate = myParam.learning_rate/minibatch_size;
            //first accumulating the gradient updates for U
            //and then updating U
            for (int i = 0;i< minibatch_size;i++)
            {
                for (int index = 0;index < myParam.ngram_size;index++)
                {
                    if (myParam.use_momentum == 1)
                    {
                        
                        for (int col = 0;col<myParam.embedding_dimension;col++)
                        {
                            U_grad.col(index*myParam.embedding_dimension + col) += adjusted_learning_rate*
                                          (W(positive_v_minibatch(i+minibatch_start_index,index),col)*
                                          positive_h_probs_minibatch.row(i).transpose()-W(negative_v_minibatch(i,index),col)*
                                          negative_h_probs_minibatch.row(i).transpose() - 2*myParam.L2_reg*U.col(index*myParam.embedding_dimension + col));
                        }
                        
                        
                    }
                    else
                    {
                        //cerr<<"no momentum"<<endl;
                        for (int col = 0;col<myParam.embedding_dimension;col++)
                        {
                            U.col(index*myParam.embedding_dimension + col) += adjusted_learning_rate*
                                          (W(positive_v_minibatch(i+minibatch_start_index,index),col)*
                                          positive_h_probs_minibatch.row(i).transpose()-W(negative_v_minibatch(i,index),col)*
                                          negative_h_probs_minibatch.row(i).transpose() - 2*myParam.L2_reg*U.col(index*myParam.embedding_dimension + col));
                        }
                    }

                }
            }
            //now updating the U's and the velocities
            if (myParam.use_momentum == 1)
            {
                for (int index = 0;index< myParam.ngram_size;index++)
                {
                    for (int col = 0;col<myParam.embedding_dimension;col++)
                    {

                        velocity_U.col(index*myParam.embedding_dimension + col) = current_momentum* velocity_U.col(index*myParam.embedding_dimension + col) +
                                                                                  U_grad.col(index*myParam.embedding_dimension + col);
                        U.col(index*myParam.embedding_dimension + col) +=  velocity_U.col(index*myParam.embedding_dimension + col);
                    }
                }
                
                
            }
            for (int i = 0;i<myParam.n_threads;i++)
            {
                  U_vector[i] = U;
            }

            //cerr<<"adjusted learning rate is "<<adjusted_learning_rate<<endl;
            for (int i = 0;i< minibatch_size;i++)
            {
                //summing up over each word in the training ngram to get the total logit
                for (int index = 0;index < myParam.ngram_size;index++)
                {
                    W.row(positive_v_minibatch(i+minibatch_start_index,index)) += adjusted_learning_rate * 
                                                                        (positive_h_probs_minibatch.row(i)*
                                                                        U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension));
                    W.row(negative_v_minibatch(i,index)) -= adjusted_learning_rate * 
                                                                        (negative_h_probs_minibatch.row(i)*
                                                                        U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension));

                    //cerr<<"W row after is "<< W.row(minibatch_v_negative(i,index))<<endl;
                    //getchar();
                    v_bias(positive_v_minibatch(i+minibatch_start_index,index)) += adjusted_learning_rate;
                    v_bias(negative_v_minibatch(i,index)) -= adjusted_learning_rate;
                    
                    //copying the updated W and v into the thread parameters. This is cheaper since we're only updating the required ones                    
                    for (int thread_id = 0;thread_id<myParam.n_threads;thread_id++)
                    {
                          W_vector[thread_id].row(positive_v_minibatch(i+minibatch_start_index,index)) = W.row(positive_v_minibatch(i+minibatch_start_index,index));
                          W_vector[thread_id].row(negative_v_minibatch(i,index)) = W.row(negative_v_minibatch(i,index));
                          v_bias_vector[thread_id](positive_v_minibatch(i+minibatch_start_index,index)) = v_bias(positive_v_minibatch(i+minibatch_start_index,index));
                          v_bias_vector[thread_id](negative_v_minibatch(i,index)) = v_bias(negative_v_minibatch(i,index));
                          //cerr<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
                    }


                }
                h_bias += adjusted_learning_rate*(positive_h_probs_minibatch.row(i)-negative_h_probs_minibatch.row(i)) ;

            }
            }
            //copying the updated h biases into the parameters
            for (int thread_id = 0;thread_id<myParam.n_threads;thread_id++)
            {
                  h_bias_vector[thread_id] = h_bias;
                  //cerr<<"h bias element "<<i<<" is "<<h_bias_vector[i]<<endl;
            }
        }

        inline double computeFreeEnergy(Matrix<int,Dynamic,Dynamic> &data,int row,param &myParam)
        {
            //cerr<<"in compute free energy"<<endl;
            Real free_energy = 0.;
            Real sum_biases = 0.;
            for (int i = 0;i<myParam.ngram_size;i++)
            {
                sum_biases += v_bias(data(row,i));
            }
            //cerr<<"sum biases is "<<sum_biases<<endl;
            Matrix<double,1,Dynamic> temp_w_sum;
            for (int h_j = 0;h_j<myParam.n_hidden;h_j++)
            {
                /*
                temp_w_sum.setZero(myParam.embedding_dimension);
                for (int index = 0;index<myParam.ngram_size;index++)
                {   
                    temp_w_sum += W.row(data(row,index));
                }
                */
                double dot_product = 0.0;
                //weight_type h_bias_sum = 0. 
                for (int index = 0;index<myParam.ngram_size;index++)
                {   
                    dot_product += U.row(h_j).block(0,index*myParam.embedding_dimension,1,myParam.embedding_dimension).dot(W.row(data(row,index))) ;
                }
                free_energy -= std::log(1 + std::exp(dot_product + h_bias(h_j)));

                //cerr<<" we are here "<<endl;
                //cerr<<"the dimesion of u row is "<<U.row(h_j).cols()<<endl;
                //free_energy -= log(1 + exp(U.row(h_j)*temp_w_sum.transpose() + h_bias(h_j))) ;
                //free_energy -= log(1 + exp(U.block(0,index*myParam.embedding_dimension,myParam.n_hidden,myParam.embedding_dimension).row(h_j).dot(
                //                          temp_w_sum.transpose()) + h_bias(h_j)));
                //cerr<<"current free energy validation is "<<free_energy_validation<<endl;
            }

            free_energy-= sum_biases ;
            //cerr<<"free energy is "<<free_energy<<endl;
            return free_energy;

        }

        //computes the free energy ratio between validation set and subset_of the training_set
        double freeEnergyRatio(Matrix<int,Dynamic,Dynamic> &validation_set,Matrix<int,Dynamic,Dynamic> &training_data,int training_validation_set_start_index,int training_validation_set_end_index,param &myParam)
        {
            double free_energy_validation,free_energy_training_validation;
            free_energy_validation = free_energy_training_validation = 0.0;
            for (int t = 0;t<validation_set.rows();t++)
            {
                free_energy_validation += computeFreeEnergy(validation_set,t,myParam);
            }
            //now computing the free energy of the training validation set 
            for (int t = training_validation_set_start_index;t<training_validation_set_end_index;t++)
            {
                free_energy_training_validation += computeFreeEnergy(training_data,t,myParam);
            }
            //cerr<<"the free energy validation was "<<free_energy_validation<<endl;
            //cerr<<"the free energy training validation was "<<free_energy_training_validation<<endl;
            double training_validation_data_set_size = training_validation_set_end_index - training_validation_set_start_index + 1;
            double ratio = (double(1./validation_set.rows())*free_energy_validation)/((1./training_validation_data_set_size)*free_energy_training_validation);
            return(ratio);
        }

        /*
        //computes the free energy ratio between validation set and subset_of the training_set
        double freeEnergyRatio(Matrix<int,Dynamic,Dynamic> &validation_set,Matrix<int,Dynamic,Dynamic> &training_data,int training_validation_set_start_index,int training_validation_set_end_index,param &myParam)
        {
            double free_energy_validation,free_energy_training_validation;
            free_energy_validation = free_energy_training_validation = 0.0;
            for (int t = 0;t<validation_set.rows();t++)
            {
                double sum_biases = 0.;
                for (int i = 0;i<myParam.ngram_size;i++)
                {
                    sum_biases += v_bias(validation_set(t,i));
                }
                //cerr<<"sum biases is "<<sum_biases<<endl;
                Matrix<double,1,Dynamic> temp_w_sum;
                for (int h_j = 0;h_j<myParam.n_hidden;h_j++)
                {

                    temp_w_sum.setZero(myParam.embedding_dimension);
                    for (int i = 0;i<myParam.ngram_size;i++)
                    {   
                        temp_w_sum += W.row(validation_set(t,i));
                    }
                    
                    free_energy_validation -= log(1 + exp(U.row(h_j)*temp_w_sum.transpose() + h_bias(h_j))) ;
                    //cerr<<"current free energy validation is "<<free_energy_validation<<endl;
                }

                free_energy_validation -= sum_biases ;
            }
            //now computing the free energy of the training validation set 
            for (int t = training_validation_set_start_index;t<training_validation_set_end_index;t++)
            {
                double sum_biases = 0.;
                for (int i = 0;i<myParam.ngram_size;i++)
                {
                    sum_biases += v_bias(training_data(t,i));
                }
                Matrix<double,1,Dynamic> temp_w_sum;
                for (int h_j = 0;h_j<myParam.n_hidden;h_j++)
                {

                    temp_w_sum.setZero(myParam.embedding_dimension);
                    for (int i = 0;i<myParam.ngram_size;i++)
                    {   
                        temp_w_sum += W.row(training_data(t,i));
                    }
                    
                    free_energy_training_validation -= log(1 + exp(U.row(h_j)*temp_w_sum.transpose() + h_bias(h_j))) ;
                }

                free_energy_training_validation -= sum_biases ;
            }
            //cerr<<"the free energy validation was "<<free_energy_validation<<endl;
            //cerr<<"the free energy training validation was "<<free_energy_training_validation<<endl;
            double training_validation_data_set_size = training_validation_set_end_index - training_validation_set_start_index + 1;
            double ratio = (double(1./validation_set.rows())*free_energy_validation)/((1./training_validation_data_set_size)*free_energy_training_validation);
            return(ratio);
        }
        */
        //get the cross entropy on a small validation set. Wait!! this is hard becuase I cannot compute hte 
        double getCrossEntropy(Matrix<int,Dynamic,Dynamic> input,Matrix<int,Dynamic,Dynamic> prediction,param & myParam)
        {
             
        }

        double inline computeReconstructionError(Matrix<int,Dynamic,Dynamic> input,Matrix<int,Dynamic,Dynamic> prediction)
        {
            //for (int i = 0;i<
            //for i in 
        }

        // write the embeddings to the file
        void writeEmbeddings(param &myParam,int epoch,vector<string> word_list)
        {
            setprecision(16);
            stringstream ss;//create a stringstream
            ss << epoch;//add number to the stream
            //return ss.str();//return a string with the contents of the stream
          
            string output_file = myParam.embeddings_prefix+"."+ss.str();
            cerr << "Writing aligner model to file : " << output_file << endl;

            ofstream EMBEDDINGOUT;
            EMBEDDINGOUT.precision(15);
            EMBEDDINGOUT.open(output_file.c_str());
            if (! EMBEDDINGOUT)
            {
              cerr << "Error : can't write to file " << output_file << endl;
              exit(-1);
            }
            for (int row = 0;row < W.rows();row++)
            {
                EMBEDDINGOUT<<word_list[row]<<"\t";
                for (int col = 0;col < W.cols();col++)
                {
                    EMBEDDINGOUT<<W(row,col)<<"\t";
                }
                EMBEDDINGOUT<<endl;
            }

            EMBEDDINGOUT.close();
        }


        void writeParams(int epoch)
        {
            //write the U parameters
            setprecision(16);
            stringstream ss;//create a stringstream
            ss << epoch;//add number to the stream
            //return ss.str();//return a string with the contents of the stream
          
            string U_output_file = "U."+ss.str();
            cerr << "Writing U params to output_file: " << U_output_file << endl;

            ofstream UOUT;
            UOUT.precision(15);
            UOUT.open(U_output_file.c_str());
            if (! UOUT)
            {
              cerr << "Error : can't write to file " << U_output_file << endl;
              exit(-1);
            }
            for (int row = 0;row < U.rows();row++)
            {
                for (int col = 0;col < U.cols()-1;col++)
                {
                    UOUT<<U(row,col)<<"\t";
                }
                //dont want an extra tab at the end
                UOUT<<U(row,U.cols()-1);
                UOUT<<endl;
            }

            UOUT.close();
          
            //write the V bias parameters
            setprecision(16);
            //return ss.str();//return a string with the contents of the stream
          
            string v_bias_output_file = "v_bias."+ss.str();
            cerr << "Writing vbias params to output_file: " << v_bias_output_file << endl;

            ofstream VOUT;
            VOUT.precision(15);
            VOUT.open(v_bias_output_file.c_str());
            if (! VOUT)
            {
              cerr << "Error : can't write to file " << v_bias_output_file << endl;
              exit(-1);
            }
            for (int col= 0;col< v_bias.cols()-1;col++)
            {
                VOUT<<v_bias(col)<<"\t";
            }
            VOUT<<v_bias(v_bias.cols()-1);
            VOUT.close();

            //write the V bias parameters
            //setprecision(16);
            //return ss.str();//return a string with the contents of the stream
          
            string h_bias_output_file = "h_bias."+ss.str();
            cerr << "Writing vbias params to output_file: " << h_bias_output_file << endl;

            ofstream HOUT;
            HOUT.precision(15);
            HOUT.open(h_bias_output_file.c_str());
            if (! HOUT)
            {
              cerr << "Error : can't write to file " << h_bias_output_file << endl;
              exit(-1);
            }
            for (int col= 0;col< h_bias.cols()-1;col++)
            {
                HOUT<<h_bias(col)<<"\t";
            }
            HOUT<<h_bias(h_bias.cols()-1);
            HOUT.close();

        }

};
