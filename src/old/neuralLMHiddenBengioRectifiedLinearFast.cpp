#include <iostream>
#include <list>
#include <ctime>
#include <cstdio>

#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

#include "param.h"
#include "neuralClasses.h"
#include "graphClasses.h"
#include "util.h"
#include "RBMDahlFunctions.h"
#include "log_add.h"
#include <cmath>
#include <stdlib.h>

typedef Node <Word_embeddings> word_node;
typedef Node <Context_matrix> context_node;
typedef Node <Hidden_layer> hidden_node;

#include "lossFunctions.h"


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
#include <omp.h>
#include <math.h>
#include <boost/unordered_map.hpp> 
#include <boost/functional.hpp>
#include "gradientCheck.h"

typedef boost::unordered_map<vector<int>, double> vector_map;
typedef boost::unordered_map<vector<int>, vector<double> > vector_vector_map;
typedef boost::unordered_map<int,vector_map > thread_vector_map;


//#define EIGEN_DONT_PARALLELIZE

using namespace std;
using namespace TCLAP;
using namespace Eigen;
using namespace boost::random;


//FUNCTION DECLARATIONS
void inline fProp(param & ,int ,int ,word_node  &,context_node  &,hidden_node &,vector<Matrix<int,Dynamic,1> >&);
void inline fPropValidation(param & ,int ,int ,word_node  &,context_node  &,hidden_node &, vector<Matrix<int,Dynamic,1> >&);

inline void printTimeStamp(time_t startT)
{
    time_t stopT, diffT;

    time(&stopT);
    diffT = difftime(stopT, startT);
    cout << endl << "Time stamp: " << ctime(&stopT);
    cout << diffT << " seconds after started\n\n";
}

int main(int argc, char** argv)
{ 
    //srand48(time(0));

    srand48(1234);
    //omp_set_num_threads(4);
    //cout<<"the number of threads is "<<omp_get_num_threads()<<endl;

    Eigen::initParallel();
	  unsigned seed = std::time(0);
    //unsigned test_seed = 1234; //for testing only
    mt19937 eng_int (seed);  // mt19937 is a standard mersenne_twister_engine, I'll pass this to sample h given v
    mt19937 eng_real (seed);  // mt19937 is a standard mersenne_twister_engine


    param myParam;
    try{
      // program options //
      CmdLine cmd("Command description message ", ' ' , "1.0");

      ValueArg<string> train_file("", "train_file", "training file" , true, "string", "string", cmd);
      ValueArg<string> words_file("", "words_file", "words file" , true, "string", "string", cmd);
      ValueArg<string> validation_file("", "validation_file", "validation file" , true, "string", "string", cmd);
      ValueArg<string> unigram_probs_file("", "unigram_probs_file", "unigram probs file" , true, "string", "string", cmd);
      ValueArg<string> embeddings_prefix("", "embeddings_prefix", "embedding prefix for the embeddings. Default:embeddings.cpp.epoch" , false, "embeddings.cpp.epoch", "string", cmd);

      ValueArg<int> ngram_size("", "ngram_size", "The size of ngram that you want to consider. Default:3", false, 3, "int", cmd);

      ValueArg<int> n_vocab("", "n_vocab", "The vocabulary size. This has to be supplied by the user", true, 1000, "int", cmd);

      ValueArg<int> n_hidden("", "n_hidden", "The number of hidden nodes. Default 100", false, 100, "int", cmd);
      ValueArg<int> n_threads("", "n_threads", "The number of threads. Default:2", false, 2, "int", cmd);

      ValueArg<int> num_noise_samples("", "num_noise_samples", "The number of noise samples. Default:25", false, 25, "int", cmd);


      ValueArg<int> embedding_dimension("", "embedding_dimension", "The size of the embedding dimension. Default 50", false, 10, "int", cmd);

      ValueArg<int> minibatch_size("", "minibatch_size", "The minibatch size. Default: 64", false, 64, "int", cmd);
      ValueArg<int> validation_minibatch_size("", "validation_minibatch_size", "The validation set minibatch size. Default: 64", false, 64, "int", cmd);
      
      ValueArg<int> num_epochs("", "num_epochs", "The number of epochs. Default:10 ", false, 10, "int", cmd);

      ValueArg<double> learning_rate("", "learning_rate", "Learning rate for training. Default:0.01", false, 0.01, "double", cmd);

      ValueArg<double> normalization_init("", "normalization_init", "The initial normalization parameter. Default:8.43385", false, 8.43385, "double", cmd);


      ValueArg<bool> use_momentum("", "use_momentum", "Use momentum during training or not. Default:0", false, 0, "bool", cmd);
      ValueArg<double> initial_momentum("", "initial_momentum", ".Initial value of momentum. Default:0.9", false, 0.9, "double", cmd);
      ValueArg<double> final_momentum("", "final_momentum", "Final value of momentum. Default:0.9", false, 0.9, "double", cmd);

      ValueArg<bool> persistent("", "persistent", "Use persistent CD or not. Default:0", false, 0, "bool", cmd);

      ValueArg<double> L2_reg("", "L2_reg", "The L2 regularization weight. Weight decay is only applied to the U parameter. Default:0.00001", false, 0.00001, "double", cmd);

      ValueArg<bool> init_normal("", "init_normal", "Initialize parameters form a normal distribution Default:0", false, 0, "bool", cmd);
      cmd.parse(argc, argv);

      // define program parameters //
      myParam.train_file = train_file.getValue();
      myParam.validation_file= validation_file.getValue();
      myParam.unigram_probs_file= unigram_probs_file.getValue();
      myParam.ngram_size = ngram_size.getValue();
      myParam.n_vocab= n_vocab.getValue();
      myParam.n_hidden= n_hidden.getValue();
      myParam.n_threads  = n_threads.getValue();
      myParam.num_noise_samples = num_noise_samples.getValue();
      myParam.embedding_dimension = embedding_dimension.getValue();
      myParam.minibatch_size = minibatch_size.getValue();
      myParam.validation_minibatch_size = validation_minibatch_size.getValue();
      myParam.num_epochs= num_epochs.getValue();
      myParam.persistent = persistent.getValue();
      myParam.learning_rate = learning_rate.getValue();
      myParam.use_momentum = use_momentum.getValue();
      myParam.embeddings_prefix = embeddings_prefix.getValue();
      myParam.words_file = words_file.getValue();
      myParam.initial_momentum = initial_momentum.getValue();
      myParam.final_momentum = final_momentum.getValue();
      myParam.L2_reg = L2_reg.getValue();
      myParam.init_normal= init_normal.getValue();
      myParam.normalization_init = normalization_init.getValue();

      // print program command to stdout//

      cout << "Command line: " << endl;

      for (int i = 0; i < argc; i++)
      {
        cout << argv[i] << endl;
      }
      cout << endl;

      cout << train_file.getDescription() << " : " << train_file.getValue() << endl;
      cout << unigram_probs_file.getDescription() << " : " << unigram_probs_file.getValue() << endl;
      cout << ngram_size.getDescription() << " : " << ngram_size.getValue() << endl;
      cout << embedding_dimension.getDescription() << " : " << embedding_dimension.getValue() << endl;
      cout << n_hidden.getDescription() << " : " << n_hidden.getValue() << endl;
      cout << n_threads.getDescription() << " : " << n_threads.getValue() << endl;
      cout << num_noise_samples.getDescription() << " : " << num_noise_samples.getValue() << endl;
      cout << n_vocab.getDescription() << " : " << n_vocab.getValue() << endl;
      cout << minibatch_size.getDescription() << " : " << minibatch_size.getValue() << endl;
      cout << validation_minibatch_size.getDescription() << " : " << validation_minibatch_size.getValue() << endl;
      cout << num_epochs.getDescription() << " : " << num_epochs.getValue() << endl;
      cout << persistent.getDescription() << " : " << persistent.getValue() << endl;
      cout << learning_rate.getDescription() << " : " << learning_rate.getValue() << endl;
      cout << use_momentum.getDescription() << " : " << use_momentum.getValue() << endl;
      cout << words_file.getDescription() << " : " << words_file.getValue() << endl;
      cout << initial_momentum.getDescription() << " : " << initial_momentum.getValue() << endl;
      cout << final_momentum.getDescription() << " : " << final_momentum.getValue() << endl;
      cout << L2_reg.getDescription() << " : " << L2_reg.getValue() << endl;
      cout << embeddings_prefix.getDescription() << " : " << embeddings_prefix.getValue() << endl;
      cout << init_normal.getDescription() << " : " << init_normal.getValue() << endl;
      cout << normalization_init.getDescription() << " : " << normalization_init.getValue() << endl;

    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    /////////////////////////READING IN THE TRAINING AND VALIDATION DATA///////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    cout<<"train file is "<<myParam.train_file<<endl;
    vector<vector<int> > unshuffled_training_data;
    //cout<<"the size of unshuffled training data is"<<unshuffled_training_data.size()<<endl;
    readTrainFile(myParam,unshuffled_training_data);
	  int training_data_size = unshuffled_training_data.size();
    cout<<"the training data size was "<<training_data_size<<endl;
    vector<vector<int> > validation_set_vector;
    readDataFile(myParam.validation_file,myParam,validation_set_vector);
    cout<<"read the validation file"<<endl;
    int validation_set_size = validation_set_vector.size();
    //cout<<"the validation set size is "<<validation_set_size<<endl;
    //getchar();
	  //now shuffling training data
	  random_shuffle ( unshuffled_training_data.begin(), unshuffled_training_data.end() );

    //now dump the training data to a temp file
    writeTempData(unshuffled_training_data,myParam);

    //clearing the data vector so that we dont' run out of memory
    unshuffled_training_data.clear();
    vector< Matrix<int,Dynamic,1> >shuffled_training_data;
    vector< Matrix<int,Dynamic,1> >validation_set;

    for (int ngram = 0;ngram < myParam.ngram_size;ngram++)
    {
        Matrix<int,Dynamic,1> column ;
        column.setZero(training_data_size);
        shuffled_training_data.push_back(column);
    }
    if (validation_set_size > 0)
    {
        for (int ngram = 0;ngram < myParam.ngram_size;ngram++)
        {
            Matrix<int,Dynamic,1> column_validation ;
            column_validation.setZero(validation_set_vector.size());
            validation_set.push_back(column_validation);
        }

    }
    
    
    cout<<"the size of the training data matrix is "<<shuffled_training_data.size()<<endl;
    cout<<"created the training data matrix and the validation set matrix"<<endl;
    //Matrix<int,Dynamic,Dynamic>  validation_set;
    
    //I have to determine the sizes of the validation set and the training data 
    //shuffled_training_data.setZero(training_data_size,myParam.ngram_size);
    //validation_set.setZero(validation_set_vector.size(),myParam.ngram_size);

    //storing the training data in the training matrix and the validation matrix
    cout<<"storing the training data into the eigen matrix"<<endl;
    
    readTrainFileMatrix("temp.dat",myParam,shuffled_training_data);
    //readTrainFileMatrix("temp.dat",myParam,shuffled_training_data);
    cout<<"storing the validation data into the eigen matrix"<<endl;
    for (int i = 0;i<validation_set_vector.size();i++)
    {
        //string joined = boost::algorithm::join(unshuffled_training_data[i], " ");
        for (int j=0;j<myParam.ngram_size;j++)
        {
            validation_set[j](i) = validation_set_vector[i][j];
        }
    }

    validation_set_vector.clear();
    //cout<<"the training data is "<<endl;
    //cout<<shuffled_training_data<<endl;
    cout<<"the size of the training data is "<<shuffled_training_data.size()<<endl;
    //writeTempDataMatrix(shuffled_training_data, myParam);
    //exit(0);
    vector<string> word_list;
    cout<<"reading words file"<<endl;
    readWordsFile(myParam.words_file, word_list);
    cout<<" the word list had size "<<word_list.size()<<endl;

    ////////////////////////////////////////ALIAS METHOD STUFF//////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////
    
    //reading the unigram probs

    vector<double> unigram_probs = vector<double>(myParam.n_vocab);
    readUnigramProbs(myParam,unigram_probs);

    
   //now i have the unigram probs, I need to setup alias method
    vector<int> J(myParam.n_vocab,-1);
    vector<double> q(myParam.n_vocab,0.);
    setupAliasMethod(unigram_probs,J ,q,myParam.n_vocab);
    cout<<"q is "<<q.size()<<endl;
    cout<<"J is "<<J.size()<<endl;

    //for multithreading, I will make copies of q and J, 
    //unigram probs, and the random number generators 
    //one for each thread
    
    vector<vector<int> > J_vector;
    vector<vector<double> > q_vector; 
    vector<vector<double> > unigram_probs_vector;
    vector<mt19937> eng_int_vector; 
    vector<mt19937> eng_real_vector; 
    vector<uniform_int_distribution<> >unif_int_vector;
    vector<uniform_real_distribution<> >unif_real_vector;
    

    for (int i=0;i<myParam.n_threads;i++)
    {
        vector<int> temp_J = J;
        J_vector.push_back(temp_J);
        vector<double> temp_q = q;
        q_vector.push_back(temp_q);
        vector<double> temp_unigram_probs = unigram_probs;
        unigram_probs_vector.push_back(temp_unigram_probs);
        clock_t t = clock()+rand();
        mt19937 eng_int_temp (t);  
        eng_int_vector.push_back(eng_int_temp);
        uniform_int_distribution<> unif_int_temp(0, myParam.n_vocab-1);
        unif_int_vector.push_back(unif_int_temp);
        mt19937 eng_real_temp (t);  // mt19937 is a standard mersenne_twister_engine
        eng_real_vector.push_back(eng_real_temp);
        uniform_real_distribution<> unif_real_temp(0.0, 1.0);
        unif_real_vector.push_back(unif_real_temp);
        cout<<"the clock was "<<t<<endl;

    }
    //initalizing the threads again
    for (int i=0;i<myParam.n_threads;i++)
    {
        clock_t t = clock()+rand();
        eng_int_vector[i].seed(t);
        eng_real_vector[i].seed(t);
 
    }

    /////CREATING THE NEURAL NETWORK GRAPH/////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    
    //initializing the neural LM
    mt19937 init_engine(clock()+rand()); //initializing the engine
    Word_embeddings D(myParam.n_vocab,myParam.embedding_dimension,init_engine,myParam.init_normal,myParam.n_threads,myParam.ngram_size-1);
    
    Output_word_embeddings D_prime(myParam.n_vocab,myParam.n_hidden,init_engine,myParam.init_normal,myParam.n_threads);

    //creating the context nodes
    Context_matrix network_context_matrix(myParam.embedding_dimension,myParam.embedding_dimension*(myParam.ngram_size-1),init_engine,myParam.init_normal,myParam.n_threads);

    Node <Context_matrix> network_context_node(&network_context_matrix,myParam.n_hidden,myParam.minibatch_size,
                                            myParam.embedding_dimension*(myParam.ngram_size-1),myParam.minibatch_size,myParam.n_hidden,
                                            myParam.validation_minibatch_size);
    
    //creating the word nodes
    Node<Word_embeddings>network_word_node(&D,myParam.embedding_dimension*(myParam.ngram_size-1),myParam.minibatch_size,-1,-1,myParam.embedding_dimension*(myParam.ngram_size-1),myParam.validation_minibatch_size);

    //creating the hidden layer
    Hidden_layer hidden(myParam.n_hidden,init_engine,myParam.init_normal,myParam.n_threads);

    //creating a node for the hidden layer and for the hidden to output matrix
    hidden_node hidden_layer_node(&hidden,myParam.embedding_dimension,myParam.minibatch_size,myParam.n_hidden,myParam.minibatch_size
                                        ,myParam.embedding_dimension,myParam.validation_minibatch_size);


    
    ///////////////////////TRAINING THE NEURAL NETWORK////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    //testing sample h given v
    //int training_data_size = unshuffled_training_data.size();
    cout<<"the traning data size is "<<training_data_size<<endl;
    int carry_over = (training_data_size%myParam.minibatch_size == 0)? 0:1;
    int num_batches = training_data_size/myParam.minibatch_size + carry_over;
    //get the generated samples
    vector<int>random_nos_int;
    vector <double> random_nos_real; 

    //readRandomNosReal("random_nos.real.1000000" ,random_nos_real);
    //readRandomNosInt("random_nos.int" ,random_nos_int);
    int random_nos_int_counter = 0;
    int random_nos_real_counter = 0;

    //cout<<"read random nos double and int"<<endl;
    //getchar();
    cout<<"num training batches is "<<num_batches<<endl;
    //getchar();
    //performing training
    clock_t t;
    //
    double current_momentum = myParam.initial_momentum;
    double momentum_delta = (myParam.final_momentum - myParam.initial_momentum)/(myParam.num_epochs-1);
    double current_learning_rate = myParam.learning_rate;
    setprecision(15);
    double current_validation_ll = -99999999999;
    cout<<"the validation set size was "<<validation_set_size<<endl;
    int num_validation_batches = validation_set_size/myParam.validation_minibatch_size;
    cout<<"num valiation batches is "<<num_validation_batches<<endl;

    //defining a bunch of variables that will be useful for multithreading. These will be declared as firstprivate
    int validation_minibatch_size = myParam.validation_minibatch_size;
    int ngram_size = myParam.ngram_size;
    int n_vocab = myParam.n_vocab;
    int embedding_dimension = myParam.embedding_dimension;
    int minibatch_size = myParam.minibatch_size;
    int num_noise_samples = myParam.num_noise_samples;
    double normalization_init = myParam.normalization_init;

    //creating the map of the normalization constants
    //vector_map c_h;
    vector_vector_map c_h;
    vector<vector_map> thread_c_h;

    for (int thread_id =0 ;thread_id<myParam.n_threads;thread_id++)
    {
        vector_map temp;
        thread_c_h.push_back(temp);
    }

    //initialize the vector map
    for (int i=0;i<training_data_size;i++)
    {
        vector<int> context;//(ngram_size-1);

        //creating the context
        for (int word = 0;word<ngram_size-1;word++)
        {
            context.push_back(shuffled_training_data[word](i));
        }
        if (c_h.find(context) == c_h.end())
        {
            //cout<<"context cache miss"<<endl;
            vector<double> temp(2);
            temp[0] = -normalization_init;
            temp[1] = exp(temp[0]);
            //c_h[context] = -normalization_init;

            c_h[context] = temp;
            for (int thread_id =0 ;thread_id<myParam.n_threads;thread_id++)
            {
                thread_c_h[thread_id][context]  = -normalization_init;
            }

        }
 

    }

    for (int epoch = 0 ;epoch<myParam.num_epochs;epoch++)
    { 
        /*
        //every 5 epochs, print the norm of the paramters
        if (epoch %5 ==0)
        {
            cout<<"the square root of the squared norm of D_prime is "<<D_prime.W.norm()<<endl;
            cout<<"the square root of the squared norm of D is "<<D.W.norm()<<endl;
            cout<<"the square root of the squared norm of the h bias is "<<hidden.h_bias.norm()<<endl;
            for (int word = 0;word<ngram_size-1;word++)
            {
                cout<<"the square root of the squared norm of the context "<<word<<" is "<<contexts[word].U.norm()<<endl;
            }
        }
        */
        cout<<"current learning rate is "<<current_learning_rate<<endl;
        if (epoch %1 ==0 && validation_set_size > 0)
        {
            //////COMPUTING VALIDATION SET PERPLEXITY///////////////////////
            ////////////////////////////////////////////////////////////////
            //first do forward propagation and then compute the probabilities
            double log_likelihood = 0.0;
            vector_map normalization_cache;
            //double min_score = 9999999999999;

            for (int validation_batch =0;validation_batch < num_validation_batches;validation_batch++)
            {
                int validation_minibatch_start_index = myParam.validation_minibatch_size * validation_batch;
                //cout<<"validation minibatch start index is "<<validation_minibatch_start_index<<endl;
                fPropValidation(myParam,validation_minibatch_start_index,myParam.validation_minibatch_size,network_word_node,
                                network_context_node,hidden_layer_node,validation_set);

                Eigen::setNbThreads(1);
                //cout<<"validation minibatch size is "<<myParam.validation_minibatch_size<<endl;
                #pragma omp parallel  firstprivate(validation_minibatch_size,validation_minibatch_start_index,ngram_size,n_vocab,embedding_dimension)
                {
                #pragma omp for reduction(+:log_likelihood)
                for (int valid_id = 0;valid_id < validation_minibatch_size;valid_id++)
                {

                    int thread_id = omp_get_thread_num();
                    //cout<<"thread id is "<<thread_id<<endl;
                    //cout<<"valid id is "<<valid_id<<endl;
                    //cout<<"the empirical output is "<<validation_set[ngram_size-1](valid_id+validation_minibatch_start_index)<<endl;
                    //getchar();
                    int empirical_output = validation_set[ngram_size-1](valid_id+validation_minibatch_start_index);
                    vector<int> context;
                    //cout<<" context is ";
                    for ( int word = 0;word < ngram_size-1;word++)
                    {
                        context.push_back(validation_set[word](valid_id+validation_minibatch_start_index));

                    }
                    //cout<<endl;
                    //getchar();
                    bool found = 1;
                    //I want to access the dictionary only 1 thread at a time. I don't think this needs to be critical
                    //cout<<"in critical section"<<endl;
                    if (normalization_cache.find(context) == normalization_cache.end())
                    {
                        found = 0;
                    }


                    if (found == 0)
                    {
                        //the predicted embedding will be the same per context
                        Matrix<double,Dynamic,1> predicted_embedding = hidden_layer_node.fProp_validation_matrix.col(valid_id);

                        //cout<<"we had a cache miss"<<endl;
                        double empirical_output_score = 0.;
                        double normalization_constant = -99999999999;
                        //we didnt find the context in the cache so we have to compute the normalization constant
                        for (int output_word_id = 0;output_word_id < n_vocab;output_word_id ++)
                        {
                            //now computing the score of all the words
                            double score =  D_prime.W.row(output_word_id).dot(predicted_embedding) + 
                                            D_prime.b(output_word_id);
                            //cout<<" score is "<<score<<endl;
                            //cout<<"computed the score "<<endl;
                            if (empirical_output == output_word_id)
                            {
                                empirical_output_score = score;
                            }

                            //log adding the score
                            normalization_constant = Log<double>::add(normalization_constant,score);
                            //cout<<"after adding the normalization constant is "<<normalization_constant<<endl;

                        }
                        //again, since we're updating the cache, it has to be thread safe
                        #pragma omp critical
                        {
						    cout<<"Final normalization constant is "<<normalization_constant<<endl;
                            normalization_cache[context] = normalization_constant;
                        }

                        log_likelihood += empirical_output_score-normalization_constant;

                        //log_likelihood_vector[valid_id] = empirical_output_score-normalization_constant;


                        //computing the score of the actual output 
                    }
                    else
                    {
                        //cout<<"we had a cache hit and the normalization constant was "<<normalization_cache[context]<<endl;
                        Matrix<double,Dynamic,1> predicted_embedding = hidden_layer_node.fProp_validation_matrix.col(valid_id);
                        //now computing the score of all the words
                        double empirical_output_score =  D_prime.W.row(empirical_output).dot(predicted_embedding) + 
                                                          D_prime.b(empirical_output);
                        //making the code threadsafe by making this section critical since we're reading from the same data structure
                        double normalization_constant = 0.0;
                        # pragma omp critical
                        {
                            normalization_constant = normalization_cache[context];
                        }

                        log_likelihood += empirical_output_score-normalization_constant;
                        //log_likelihood_vector[valid_id] = empirical_output_score-normalization_constant;
                    }
                    
                    
                }
                }
                #pragma omp barrier

            }

            //cout<<"The min score is "<<min_score<<endl;
            cout<<"the log likelihood is "<<log_likelihood<<endl;
            //computing perplexity
            //perplexity = 
            if (log_likelihood < current_validation_ll)
            { 
                current_learning_rate /= 2; //halving the learning rate if the validation perplexity decreased
            }
            current_validation_ll = log_likelihood;
            //getchar();
            //clearing the validation matrices
            network_context_node.fProp_validation_matrix.setZero();
            network_word_node.fProp_validation_matrix.setZero();
            hidden_layer_node.fProp_validation_matrix.setZero();



        }
        else
        {
            cout<<"Validation set size is 0!"<<endl;
        }

        if (myParam.use_momentum == 0)
        {
            current_momentum = -1;
        }
        cout<<"current momentum is "<<current_momentum<<endl;
        //current_momentum  =-1;
        cout<<"epoch is "<<epoch+1<<endl;
        double average_f = 0.0;
        for(int batch=0;batch<num_batches;batch++)
        {
            //cout<<"batch number is "<<batch<<endl;
            if (batch%1000 == 0)
            {
                cout<<"processed "<<batch<<" batches"<<endl;
            } 
            //int cdk = 0; //set this for now 
            int current_minibatch_size = ((batch+1)*myParam.minibatch_size <= training_data_size)? myParam.minibatch_size:training_data_size%myParam.minibatch_size;
            int minibatch_start_index = myParam.minibatch_size * batch;
            //cout<<"minibatch start index is "<<minibatch_start_index<<endl;
            double adjusted_learning_rate = current_learning_rate/current_minibatch_size;
            //cout<<"the adjusted learning rate is "<<adjusted_learning_rate<<endl;
            /*
            if (batch == rand() % num_batches)
            {
                cout<<"we are checking the gradient in batch "<<batch<<endl;
                /////////////////////////CHECKING GRADIENTS////////////////////////////////////////
                gradientChecking(myParam,minibatch_start_index,current_minibatch_size,word_nodes,context_nodes,hidden_layer_node,hidden_layer_to_output_node,
                              shuffled_training_data,c_h,unif_real_vector,eng_real_vector,unif_int_vector,eng_int_vector,unigram_probs_vector,
                              q_vector,J_vector,D_prime);
            }
            */
            ///////////////////FORWARD PROPAGATION/////////////////////////
            ////////////////////////////////////////////////////////////////
            fProp(myParam,minibatch_start_index,current_minibatch_size,network_word_node,network_context_node,hidden_layer_node,shuffled_training_data);
            /*
            if (epoch%2==0 && epoch>=1)
            {
                cout<<"the hidden node activations are "<<hidden_layer_node.fProp_matrix<<endl;
                getchar();
            }
            */

            /////////////////COMPUTING THE NCE LOSS FUNCTION//////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////
            //////////////////BACK PROPAGATION//////////////////////////////////
            ////////////////////////////////////////////////////////////////
            //computing the loss function 
            //now computing the loss function
            //Matrix<double,Dynamic,Dynamic> output_gradient;
            Matrix<double,Dynamic,Dynamic> minibatch_predicted_embeddings(myParam.embedding_dimension,myParam.minibatch_size);
            Matrix<double,Dynamic,1> minibatch_positive_weights(myParam.minibatch_size);
            Matrix<double,Dynamic,Dynamic> minibatch_negative_weights(myParam.minibatch_size,myParam.num_noise_samples);
            Matrix<int,Dynamic,Dynamic> minibatch_negative_samples(myParam.minibatch_size,myParam.num_noise_samples);

            //for each thread, storing the training data locations
            
            //output_gradient.setZero(myParam.embedding_dimension,current_minibatch_size);
            

            //int thread_id = 0; //for now, this is a proxy. When I'm multithreading this code, the thread ID will change
            //creating the unordered map for each thread
            //thread_vector_map c_h_gradient;
            vector<vector_map> c_h_gradient;
            for (int thread_id =0 ;thread_id<myParam.n_threads;thread_id++)
            {
                vector_map temp;
                c_h_gradient.push_back(temp);
            }
            clock_t t;
            //c_h_gradient += minibatch_positive_weights(train_id)
            //parallelizing the creation with multithreading
            //omp_set_num_threads(myParam.n_threads);
            //Eigen::initParallel();
            Eigen::setNbThreads(1);
            t = clock();
            #pragma omp parallel firstprivate(minibatch_size,minibatch_start_index,ngram_size,n_vocab,embedding_dimension, \
                                    num_noise_samples,normalization_init)
            {
              /*
              #pragma omp master
              {
                  cout<<"num threads is "<<omp_get_num_threads()<<endl;
              }
              */
              //cout<<"thread id is "<<thread_id<<endl;

            #pragma omp for reduction(+:average_f) schedule(static,100)
            for (int train_id = 0;train_id < minibatch_size;train_id++)
            {
                //clock_t t = clock();
                //double sample_f = 0.;
                int thread_id = omp_get_thread_num();
                int output_word = shuffled_training_data[ngram_size-1](minibatch_start_index+train_id);
                //int output_word = (*thread_data_col_locations[thread_id][ngram_size-1])(minibatch_start_index+train_id);
                //cout<<"output word is "<<output_word<<endl;
                Matrix<double,Dynamic,1> predicted_embedding=hidden_layer_node.fProp_matrix.col(train_id);
                
                vector<int> context;//(ngram_size-1);

                //creating the context
                for (int word = 0;word<ngram_size-1;word++)
                {
                    context.push_back(shuffled_training_data[word](minibatch_start_index+train_id));
                    //context.push_back((*thread_data_col_locations[thread_id][word])(minibatch_start_index+train_id));
                }

                //double log_inv_normalization_const_h = 0.;
                //getting a normalization constant and making it threasafe
                
                //this region does not need to be critical because its just a read
                //log_inv_normalization_const_h = c_h[context];


                //double inv_normalization_const_h = exp(log_inv_normalization_const_h);

                double inv_normalization_const_h = c_h[context][1];
                //cout<<"The normalization constant is "<<inv_normalization_const_h<<endl;
                //setting the gradient for that context to 0;
                //double c_h_gradient = 0.0;

                minibatch_predicted_embeddings.col(train_id) = predicted_embedding;
                double score = D_prime.W.row(output_word).dot(predicted_embedding) + D_prime.b(output_word);
                double unnorm_positive_prob = exp(score);
                double positive_prob = unnorm_positive_prob*inv_normalization_const_h;
                //cout<<"unnorm positive prob was "<<unnorm_positive_prob<<endl;
                minibatch_positive_weights(train_id) = num_noise_samples*unigram_probs_vector[thread_id][output_word]/
                                                       (unnorm_positive_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][output_word]) ;
                //cout<<"the score was "<<score<<endl;
                //cout<<"the positive weight was "<< minibatch_positive_weights(train_id)<<endl;

                //sample_f += log(positive_prob/(positive_prob + num_noise_samples * unigram_probs_vector[thread_id][output_word]));
                
                if (c_h_gradient[thread_id].find(context) == c_h_gradient[thread_id].end())
                {
                    c_h_gradient[thread_id][context] = minibatch_positive_weights(train_id);
                }
                else
                {
                    //cout<<"we got a repeat!"<<endl;
                    c_h_gradient[thread_id][context] += minibatch_positive_weights(train_id);
                }

                
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
                        sample = mixture_component;
                    }
                    else
                    {
                        sample = J_vector[thread_id][mixture_component];
                    }

                    //vector<int> context(ngram_size-1);

                    //cout<<"the sample was "<<sample<<endl;
                    assert (sample >= 0);
                    minibatch_negative_samples(train_id,sample_id) = sample;
                    double negative_score = D_prime.W.row(sample).dot(predicted_embedding) + D_prime.b(sample);
                    double negative_unnorm_prob = exp(negative_score);
                    double negative_prob = negative_unnorm_prob*inv_normalization_const_h;
                    minibatch_negative_weights(train_id,sample_id) = negative_unnorm_prob*inv_normalization_const_h/
                                                                    (negative_unnorm_prob*inv_normalization_const_h + num_noise_samples * unigram_probs_vector[thread_id][sample]);
                    //sample_f += log(num_noise_samples * unigram_probs_vector[thread_id][sample]/( num_noise_samples * unigram_probs_vector[thread_id][sample] + negative_prob));
                    c_h_gradient[thread_id][context] -= minibatch_negative_weights(train_id,sample_id);
                }
                //average_f += sample_f;

            }
            //getchar();
            }
            #pragma omp barrier
            //cout<<"the time taken to compute the noise samples was "<<clock()-t<<endl;
            //updating the normalization constants
            for (int thread_id=0;thread_id<myParam.n_threads;thread_id++)
            {
                vector_map::iterator it;
                for (it = c_h_gradient[thread_id].begin();it != c_h_gradient[thread_id].end();it++)
                {
                    c_h[(*it).first][0] += adjusted_learning_rate * (*it).second;
                    c_h[(*it).first][1] = exp(c_h[(*it).first][0]);

                }
            }

            //cout<<"the time taken to update normalization constants was "<<clock()-t<<endl;
            //t = clock();
            //first comput the backprop gradient
            Matrix<double,Dynamic,Dynamic> context_bProp_matrix;//(myParam.embedding_dimension,current_minibatch_size);
            context_bProp_matrix.setZero(myParam.n_hidden,current_minibatch_size);
            D_prime.bProp(shuffled_training_data[myParam.ngram_size-1],minibatch_positive_weights,
                            minibatch_negative_samples,minibatch_negative_weights,
                            context_bProp_matrix,minibatch_start_index,current_minibatch_size,myParam.num_noise_samples);
            
            D_prime.computeGradient(minibatch_predicted_embeddings,shuffled_training_data[myParam.ngram_size-1],minibatch_positive_weights,
                            minibatch_negative_samples,minibatch_negative_weights,
                            minibatch_start_index,current_minibatch_size,myParam.num_noise_samples,adjusted_learning_rate,current_momentum);

            //now doing backprop on the hidden node
            hidden_layer_node.param->bPropRectifiedLinear(context_bProp_matrix,hidden_layer_node.bProp_matrix,hidden_layer_node.fProp_matrix);
            
            //now doing backprop on the context matrices
            network_context_node.param->bProp(hidden_layer_node.bProp_matrix,network_context_node.bProp_matrix);
            //cout<<"the time taken to do bprop on the context matrix is "<<clock()-t<<endl;
            //t = clock();
            network_context_node.param->computeGradientOmp(hidden_layer_node.bProp_matrix,network_word_node.fProp_matrix,adjusted_learning_rate,current_momentum,myParam.L2_reg);

            network_word_node.param->computeGradient(network_context_node.bProp_matrix,shuffled_training_data,minibatch_start_index,
                                                        current_minibatch_size,adjusted_learning_rate,current_momentum,myParam.L2_reg,myParam.ngram_size-1);

            network_context_node.fProp_matrix.setZero();
            network_context_node.bProp_matrix.setZero();
            network_word_node.fProp_matrix.setZero();
            hidden_layer_node.fProp_matrix.setZero();
            hidden_layer_node.bProp_matrix.setZero();
            //network_context_node.fProp_validation_matrix.setZero();
            //network_word_node.fProp_validation_matrix.setZero();
            //hidden_layer_node.fProp_validation_matrix.setZero();


        }
        //cout<<"the average f for the training data was "<<average_f/training_data_size<<endl;
        current_momentum += momentum_delta;
        //writing the parameters after every epoch  
        stringstream ss;//create a stringstream
        ss << epoch;//add number to the stream
        //return ss.str();//return a string with the contents of the stream
        cout<<"..Writing the parameters.."<<endl;
        //first write the output word representations and bias
        string D_prime_W_output_file = "D_prime_W." + ss.str();
        writeMatrix(D_prime.W,D_prime_W_output_file);
        string D_prime_b_output_file = "D_prime_b." + ss.str();
        writeVector(D_prime.b,D_prime_b_output_file);

        //then writing the input word embeddings
        string D_W_output_file = "D_W." + ss.str();
        writeMatrix(D.W,D_W_output_file);
        string U_output_file = "U."+ss.str();
        writeMatrix(network_context_matrix.U,U_output_file);
    }
    return 0;
}

void inline fProp(param & myParam,int minibatch_start_index,int current_minibatch_size,word_node &network_word_node,
            context_node &network_context_node,hidden_node &hidden_layer_node,vector<Matrix<int,Dynamic,1> >&data)
{
    /////FORWARD PROPAGATION/////////////
    //doing forward propagation first with word nodes
    omp_set_num_threads(1);
    //clock_t t = clock();
    network_word_node.param->fPropOmp(data,network_word_node.fProp_matrix,minibatch_start_index,current_minibatch_size,myParam.ngram_size-1);
    //cout<<"time for fprop for the word embeddings was "<<clock()-t<<endl;

    omp_set_num_threads(myParam.n_threads);
    //Eigen::setNbThreads(myParam.n_threads);
    //t = clock();
    network_context_node.param->fProp(network_word_node.fProp_matrix,network_context_node.fProp_matrix);
    //cout<<"time for fprop for the context matrix was "<<clock()-t<<endl;

    //doing forward prop with the hidden nodes
    hidden_layer_node.param->fPropRectifiedLinear(network_context_node.fProp_matrix,hidden_layer_node.fProp_matrix);


}

void inline fPropValidation(param & myParam,int start_index,int validation_set_size,word_node &network_word_node,
                           context_node &network_context_node,hidden_node &hidden_layer_node, vector<Matrix<int,Dynamic,1> >&data)
{
    /////FORWARD PROPAGATION/////////////
    omp_set_num_threads(1);
    network_word_node.param->fPropOmp(data,network_word_node.fProp_validation_matrix,start_index,validation_set_size,myParam.ngram_size-1);
    //Eigen::setNbThreads(myParam.n_threads);
    omp_set_num_threads(myParam.n_threads);
    network_context_node.param->fProp(network_word_node.fProp_validation_matrix,network_context_node.fProp_validation_matrix);
   
    //doing forward prop with the hidden nodes
    hidden_layer_node.param->fPropRectifiedLinear(network_context_node.fProp_validation_matrix,hidden_layer_node.fProp_validation_matrix);

}




