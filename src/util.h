#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <math.h>
#include <algorithm>

#include <boost/unordered_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/functional/hash.hpp>
#ifdef USE_CHRONO
#include <boost/chrono.hpp>
#endif

#include <Eigen/Dense>

#include "define.h"
#include "maybe_omp.h"
#include "vocabulary.h"

// Make matrices hashable

typedef long long int data_size_t; // training data can easily exceed 2G instances

/*
namespace Eigen {
    template <typename Derived>
    size_t hash_value(const DenseBase<Derived> &m)
    {
        size_t h=0;
	for (int i=0; i<m.rows(); i++)
	    for (int j=0; j<m.cols(); j++)
	        boost::hash_combine(h, m(i,j));
	return h;
    }
}
*/

namespace nplm
{
	
struct gradClipper{
  precision_type operator() (precision_type x) const { 
    return std::min(1., std::max(double(x),-1.));
    //return(x);
  }
};

typedef boost::unordered_map<int,bool> int_map;

void splitBySpace(const std::string &line, std::vector<std::string> &items);
void readWordsFile(std::ifstream &TRAININ, std::vector<std::string> &word_list);
void readWordsFile(const std::string &file, std::vector<std::string> &word_list);
void writeWordsFile(const std::vector<std::string> &words, std::ofstream &file);
void writeWordsFile(const std::vector<std::string> &words, const std::string &filename);
void readDataFile(const std::string &filename, int &ngram_size, std::vector<int> &data, int minibatch_size=0);
void readUnigramProbs(const std::string &unigram_probs_file, std::vector<precision_type> &unigram_probs);
void readWeightsFile(std::ifstream &TRAININ, std::vector<float> &weights);
void readSentFile(const std::string &filename, std::vector<std::vector <int> > &data, int minibatch_size, data_size_t &num_tokens);
void miniBatchify(const std::vector<std::vector <int> > &sentences, 
				 std::vector<int > &minibatch_sentences,
				const int minibatch_start_index,
				const int minibatch_end_index,
				unsigned int &max_sent_len,
				bool is_input,
				unsigned int &minibatch_tokens);
void miniBatchifyEncoder(const std::vector<std::vector <int> > &sentences, 
				 std::vector<int > &minibatch_sentences,
				const int minibatch_start_index,
				const int minibatch_end_index,
				unsigned int &max_sent_len,
				unsigned int &minibatch_tokens,
				bool data_or_sentence_vector);
void miniBatchifyDecoder(const std::vector<std::vector <int> > &sentences, 
				 std::vector<int > &minibatch_sentences,
				const int minibatch_start_index,
				const int minibatch_end_index,
				unsigned int &max_sent_len,
				unsigned int &minibatch_tokens,
				bool data_or_sentence_vector);	
void miniBatchifyDecoder(const std::vector<std::vector <int> > &sentences, 
				 std::vector<int > &minibatch_sentences,
				const int minibatch_start_index,
				const int minibatch_end_index,
				unsigned int &max_sent_len,
				unsigned int &minibatch_tokens,
				bool data_or_sentence_vector,
				int pad_value);				
void createVocabulary(std::vector<std::vector<std::string> > &sentences, vocabulary &vocab);

void integerize(std::vector<std::vector<std::string> > &word_sentences, 
				std::vector<std::vector<int> > &int_sentences, 
				vocabulary &vocab);

void integerize(std::vector<std::vector<std::string> > &word_sentences, 
				std::vector<std::vector<int> > &int_sentences, 
				vocabulary &vocab,
				int start_index,
				int end_offset);

void buildDecoderVocab(std::vector<std::vector<std::string> > word_training_output_sent, 
						vocabulary &vocab,
						int start_index,
						int output_offset);
//for creating memory								
template<typename dType>
void allocate_Matrix_CPU(dType **h_matrix,int rows,int cols) {
	*h_matrix = (dType *)malloc(rows*cols*sizeof(dType));
}						

//Aug-27-2015
//Structure for saving decoder trace. The current decoder trace assumes so much about the LSTM model. 
//The decoder trace should contain a generic dump that that can be printed out. For that, a simple structure
//That contains the state_name and the value should be created
template<typename Derived>
struct state_storage {
	std::string state_name;
	Eigen::MatrixBase<Derived> state_value;
};

//template <typename T> readSentFile(const std::string &file, T &sentences);

////FOR GETTING THE K-BEST ITEMS in beam search
///row and col will store the row index and col idex of the k-best item
///For the probability matrix, row will also be the word id and col will correspond
/// to the previous k-best item sequence index that this k-best item is coming from


struct value_and_index {
	int value;
	int index;
};


struct beam_item {
	double value;
	int row;
	int col;
};

////FOR STORING A K-BEST SEQUENCE

struct k_best_seq_item {
	double value;
	std::vector<int> seq;
};
//I think the comparator should be more general. 
//It should be templated on a type which could be a class or a struct
//and all it needs is that it should return a value
//This comparator returns true if a>b, which means it will end up sorting in 
//non-decreasing order or it will generate a min-heap if used in a heap
template <typename item>
struct comparator {
  bool operator()(const item& a,const item& b) const{
    return a.value > b.value;
  }
};
//typedef long long int data_size_t; // training data can easily exceed 2G instances

template<typename DerivedValue> 
void getKBest(const Eigen::MatrixBase<DerivedValue> &values, 
				std::vector<beam_item>& k_best_list, 
				std::vector<k_best_seq_item>& k_best_seq_list,
				const int k){
	//UNCONST(DerivedIndex, const_indices, indices);
	//vector<
	//std::cerr<<"Values is "<<values<<std::endl;
	int rows = values.rows();
	int cols = values.cols();
	//getchar();
	//create k elements in the priority queue, 
	//vector<beam_item> k_best_list;
	int size = rows*cols;
	int current_beam_index=0;
	//std::cerr<<"k_best_seq_list.size() "<<k_best_seq_list.size()<<std::endl;
	//std::cerr<<"values.cols() "<<values.cols()<<" values.rows() "<<values.rows()<<std::endl;
	for (; current_beam_index<k && current_beam_index<size; current_beam_index++) {
		beam_item item;
		//item.value = *(values.derived().data()+k);
		item.row = current_beam_index%rows;
		item.col = current_beam_index/rows;
		item.value = values(item.row, item.col);
		//std::cerr<<"item.row "<<item.row<<std::endl;
		//std::cerr<<"item.col "<<item.col<<std::endl;
		//std::cerr<<"item.value "<<item.value<<std::endl;
		//If we are getting the initial k best list, then the k-best list will be empty.
		//Else, we should get the score of the n-1 words of the sequence (prefix ?)
		if (k_best_seq_list.size() > 0) { 
			//We have to add the probability of the previous subsequence as well
			item.value = values(item.row, item.col) + k_best_seq_list.at(item.col).value;
		}
		//std::cerr<<"inserting values("<<item.row<<","<<item.col<<") "<<values(item.row,item.col)<<std::endl;
		k_best_list.push_back(item);
		//std::cerr<<" row is "<<item.row<<" col is "<<item.col<<std::endl;
		//std::cerr<<"Value while creating the beam is "<<item.value<<std::endl;
	}
	make_heap(k_best_list.begin(), k_best_list.end(), comparator<beam_item>());
	//Now go over all the elements of the list
	for (; current_beam_index<size; current_beam_index++){
		beam_item item;
		item.row = current_beam_index%rows;
		item.col = current_beam_index/rows;
		item.value = values(item.row,item.col);
		if (k_best_seq_list.size() > 0) {
			//We have to add the probability of the previous subsequence as well
			item.value = values(item.row, item.col) + k_best_seq_list.at(item.col).value;
		}		
		//td::cerr<<"inserting values("<<item.row<<","<<item.col<<") "<<values(item.row,item.col)<<std::endl;
		k_best_list.push_back(item);
		std::pop_heap(k_best_list.begin(), k_best_list.end(), comparator<beam_item>());
		k_best_list.pop_back();
		//make_heap(k_best_list.begin(), k_best_list.end(), comparator);
	}
	std::sort_heap(k_best_list.begin(), k_best_list.end(), comparator<beam_item>());
	//k_best_list;
}

///FUNCTIONS FOR GETTING K-BEST ITEMS END HERE

//Populates the sentences into a vector of vectors.
template <typename T>
void readSentFile(const std::string &file, 
				std::vector<std::vector<T> > &sentences,
				data_size_t &tokens,
				const bool add_start_stop,
				const bool is_output)
{
	  std::cerr << "Reading sentences from: " << file << std::endl;

	  std::ifstream TRAININ;
	  TRAININ.open(file.c_str());
	  if (! TRAININ)
	  {
	    std::cerr << "Error: can't read from file " << file<< std::endl;
	    exit(-1);
	  }

	  std::string line;
	  while (getline(TRAININ, line))
	  {
	    std::vector<T> words;
	    splitBySpace(line, words);
		if (words.size() > 100){
			std::cerr<<"Error! The training sentence length was greater than 100. Maximum allowed sentence length is 100"<<std::endl;
			exit(1);
		}
		if (add_start_stop) {
			words.insert(words.begin(),"<s>");
			if (is_output){
				words.push_back("</s>");
			}
		}
	    sentences.push_back(words);
		tokens += words.size()-1;
	  }
	  
	  TRAININ.close();
}

//Populates the sentences into a vector of vectors and the offset says which line to read
template <typename T>
void readOddSentFile(const std::string &file, 
				std::vector<std::vector<T> > &sentences,
				data_size_t &tokens,
				const bool add_start_stop,
				const bool is_output,
				const bool reverse)
{
	  std::cerr << "Reading Odd sentences from: " << file << std::endl;

	  std::ifstream TRAININ;
	  TRAININ.open(file.c_str());
	  if (! TRAININ)
	  {
	    std::cerr << "Error: can't read from file " << file<< std::endl;
	    exit(-1);
	  }

	  std::string line;
	  int counter = 0;
	  while (getline(TRAININ, line))
	  {
		if (counter % 2 == 1) {
			//std::cerr<<"counter is 1"<<std::endl;
		    std::vector<T> words;
		    splitBySpace(line, words);
			if (words.size() > 100){
				std::cerr<<"Error! The training sentence length was greater than 100. Maximum allowed sentence length is 100"<<std::endl;
				exit(1);
			}
			if (reverse) { //If the user wants to reverse
				std::reverse(words.begin(),words.end());
			}
			//for (int i=0;i<words.size(); i++){
			//	std::cerr<<" words "<<i<<" "<<words.at(i);
			//}
			//std::cerr<<std::endl;
			//getchar();
			if (add_start_stop) {
				words.insert(words.begin(),"<s>");
				if (is_output){
					words.push_back("</s>");
				}
			}

		    sentences.push_back(words);
			tokens += words.size()-1;
		}
		counter++;
	  }
	  
	  TRAININ.close();
}

//Populates the sentences into a vector of vectors and the offset says which line to read
template <typename T>
void readEvenSentFile(const std::string &file, 
				std::vector<std::vector<T> > &sentences,
				data_size_t &tokens,
				const bool add_start_stop,
				const bool is_output,
				const bool reverse)
{
	  std::cerr << "Reading Even sentences from: " << file << std::endl;

	  std::ifstream TRAININ;
	  TRAININ.open(file.c_str());
	  if (! TRAININ)
	  {
	    std::cerr << "Error: can't read from file " << file<< std::endl;
	    exit(-1);
	  }

	  std::string line;
	  int counter = 0;
	  while (getline(TRAININ, line))
	  {
		if (counter % 2 == 0) {
			//std::cerr<<"counter is 0"<<std::endl;
		    std::vector<T> words;
		    splitBySpace(line, words);
			if (words.size() > 100){
				std::cerr<<"Error! The training sentence length was greater than 100. Maximum allowed sentence length is 100"<<std::endl;
				exit(1);
			}
			if (reverse) {//If the user wants to reverse
				std::reverse(words.begin(),words.end());	
			}
			//for (int i=0;i<words.size(); i++){
			//	std::cerr<<" words "<<i<<" "<<words.at(i);
			//}					
			//std::cerr<<std::endl;
			//getchar();
			if (add_start_stop) {
				words.insert(words.begin(),"<s>");
				if (is_output){
					words.push_back("</s>");
				}
			}
		    sentences.push_back(words);
			tokens += words.size();
		}
		counter++;
	  }
	  
	  TRAININ.close();
}

inline void intgerize(std::vector<std::string> &ngram,std::vector<int> &int_ngram){
        int ngram_size = ngram.size();
        for (int i=0;i<ngram_size;i++)
        int_ngram.push_back(boost::lexical_cast<int>(ngram[i]));
}

// Functions that take non-const matrices as arguments
// are supposed to declare them const and then use this
// to cast away constness.
#define UNCONST(t,c,uc) Eigen::MatrixBase<t> &uc = const_cast<Eigen::MatrixBase<t>&>(c);

template <typename Derived>
void initMatrix(boost::random::mt19937 &engine,
		const Eigen::MatrixBase<Derived> &p_const,
		bool init_normal, precision_type range)
{
    UNCONST(Derived, p_const, p);
    if (init_normal == 0)
     // initialize with uniform distribution in [-range, range]
    {
        boost::random::uniform_real_distribution<> unif_real(-range, range); 
        for (int i = 0; i < p.rows(); i++)
        {
            for (int j = 0; j< p.cols(); j++)
            {
                p(i,j) = unif_real(engine);  
				//p(i,j) = 1.; 
            }
        }

    }
    else 
      // initialize with gaussian distribution with mean 0 and stdev range
    {
        boost::random::normal_distribution<precision_type> unif_normal(0., range);
        for (int i = 0; i < p.rows(); i++)
        {
            for (int j = 0; j < p.cols(); j++)
            {
                p(i,j) = unif_normal(engine);    
            }
        }
    }
}

template<typename Derived>
void scaleAndNormClip(const Eigen::MatrixBase<Derived> &const_param,
					 std::vector<int> &update_items,
					 int current_minibatch_size,
					 precision_type norm_threshold){
	UNCONST(Derived, const_param, param);
	int num_items = update_items.size();
	precision_type squared_param_norm = 0.;
    for (int item_id=0; item_id<num_items; item_id++)
    {
        int update_item = update_items[item_id];
		param.row(update_item) /= current_minibatch_size;
		squared_param_norm += param.row(update_item).squaredNorm();
	}
	
	precision_type param_norm = sqrt(squared_param_norm);
	//std::cerr<<"sparse param_norm "<<param_norm<<std::endl;
	if (param_norm > norm_threshold){
		//param *= norm_threshold/param_norm;
	    for (int item_id=0; item_id<num_items; item_id++)
	    {
	        int update_item = update_items[item_id];
			param.row(update_item) *= norm_threshold/param_norm;
			
		}
	}
	squared_param_norm = 0.;
    for (int item_id=0; item_id<num_items; item_id++)
    {
        int update_item = update_items[item_id];
		//param.row(update_item) /= current_minibatch_size;
		squared_param_norm += param.row(update_item).squaredNorm();
	}
		
	param_norm = sqrt(squared_param_norm);
	//std::cerr<<"new sparse param_norm "<<param_norm<<std::endl;
}


template<typename Derived>
void scaleAndNormClip(const Eigen::MatrixBase<Derived> &const_param,
					 int current_minibatch_size,
					 precision_type norm_threshold){
	UNCONST(Derived, const_param, param);
	param /= current_minibatch_size;
	precision_type param_norm = param.norm();
	//std::cerr<<"param_norm "<<param_norm<<std::endl;
	if (param_norm > norm_threshold){
		param *= norm_threshold/param_norm;
	}
}

//Change a random position in the parameter by an offset. this is used for gradient checking
template<typename Derived>
void changeRandomParamInMatrix(const Eigen::MatrixBase<Derived> &const_param, 
		precision_type offset,
		int &rand_row,
		int &rand_col) {
	UNCONST(Derived, const_param, param);
	if (rand_row == -1 && rand_col == -1) {
		int num_rows = param.rows();
		int num_cols = param.cols();
		rand_row = rand() % num_rows;
		rand_col = rand() % num_cols;
	}
	param(rand_row,rand_col) += offset;
	
}

template <typename Derived>
void initBias(boost::random::mt19937 &engine,
		const Eigen::MatrixBase<Derived> &p_const,
		bool init_normal, precision_type range)
{
    UNCONST(Derived, p_const, p);
    if (init_normal == 0)
     // initialize with uniform distribution in [-range, range]
    {
        boost::random::uniform_real_distribution<> unif_real(-range, range); 
        for (int i = 0; i < p.size(); i++)
        {
            p(i) = unif_real(engine);
			//p(i) = 1;  
        }

    }
    else 
      // initialize with gaussian distribution with mean 0 and stdev range
    {
        boost::random::normal_distribution<precision_type> unif_normal(0., range);
        for (int i = 0; i < p.size(); i++)
        {
            p(i) = unif_normal(engine);    
        }
    }
}



	
template <typename Derived>
void readMatrix(std::ifstream &TRAININ, Eigen::MatrixBase<Derived> &param_const)
{
    UNCONST(Derived, param_const, param);

    int i = 0;
    std::string line;
    std::vector<std::string> fields;
    
    while (std::getline(TRAININ, line) && line != "")
    {
        splitBySpace(line, fields);
	if (fields.size() != param.cols())
	{
	    std::ostringstream err;
	    err << "error: wrong number of columns (expected " << param.cols() << ", found " << fields.size() << ")";
	    throw std::runtime_error(err.str());
	}
	
	if (i >= param.rows())
	{
	    std::ostringstream err;
	    err << "error: wrong number of rows (expected " << param.rows() << ", found " << i << ")";
	    throw std::runtime_error(err.str());
	}
	
	for (int j=0; j<fields.size(); j++)
	{
	    param(i,j) = boost::lexical_cast<typename Derived::Scalar>(fields[j]);
	}
	i++;
    }
    
    if (i != param.rows())
    {
        std::ostringstream err;
	err << "error: wrong number of rows (expected " << param.rows() << ", found more)";
	throw std::runtime_error(err.str());
    }
}

template <typename Derived>
void readMatrix(const std::string &param_file, const Eigen::MatrixBase<Derived> &param_const)
{
    UNCONST(Derived, param_const, param);
    std::cerr << "Reading data from file: " << param_file << std::endl;
    
    std::ifstream TRAININ(param_file.c_str());
    if (!TRAININ)
    {
        std::cerr << "Error: can't read training data from file " << param_file << std::endl;
	exit(-1);
    }
    readMatrix(TRAININ, param);
    TRAININ.close();
}

template <typename Derived>
void readMatrix(const Eigen::MatrixBase<Derived> &param, const std::string &filename)
{
    std::cerr << "Writing parameters to " << filename << std::endl;

    std::ofstream OUT;
    //OUT.precision(16);
    OUT.open(filename.c_str());
    if (! OUT)
    {
      std::cerr << "Error: can't write to file " << filename<< std::endl;
      exit(-1);
    }
    writeMatrix(param, OUT);
    OUT.close();
}


template <typename Derived>
void writeMatrix(const Eigen::MatrixBase<Derived> &param, std::ofstream &OUT)
{
	
    for (int row = 0;row < param.rows();row++)
    {
        int col;
        for (col = 0;col < param.cols()-1;col++)
        {
            OUT<<param(row,col)<<"\t";
        }
        //dont want an extra tab at the end
        OUT<<param(row,col)<<std::endl;
    }
}

template <typename Derived>
double logsum(const Eigen::MatrixBase<Derived> &v)
{
    int mi; 
    precision_type m = v.maxCoeff(&mi);
    double logz = 0.0;
    for (int i=0; i<v.rows(); i++)
        if (i != mi)
	    logz += std::exp(double(v(i) - m));
    logz = log1p(logz) + m;
    return logz;
}

double logadd(double x, double y);

#ifdef USE_CHRONO
class Timer 
{
    typedef boost::chrono::high_resolution_clock clock_type;
    typedef clock_type::time_point time_type;
    typedef clock_type::duration duration_type;
    std::vector<time_type> m_start;
    std::vector<duration_type> m_total;
public:
    Timer() { }
    Timer(int n) { resize(n); }
    void resize(int n) { m_start.resize(n); m_total.resize(n); }
    int size() const { return m_start.size(); }
    void start(int i);
    void stop(int i);
    void reset(int i);
    precision_type get(int i) const;
};

extern Timer timer;
#define start_timer(x) timer.start(x)
#define stop_timer(x) timer.stop(x)
#else
#define start_timer(x) 0
#define stop_timer(x) 0
#endif

int setup_threads(int n_threads);

} // namespace nplm
