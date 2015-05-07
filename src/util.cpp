#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <deque>
#include <vector>

#include <boost/unordered_map.hpp> 
#include <boost/algorithm/string.hpp>

#include "maybe_omp.h"
#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include "util.h"

extern double drand48();

using namespace Eigen;
using namespace std;
using namespace boost::random;

namespace nplm
{

void splitBySpace(const std::string &line, std::vector<std::string> &items)
{
    string copy(line);
    boost::trim_if(copy, boost::is_any_of(" \t"));
    if (copy == "")
    {
	items.clear();
	return;
    }
    boost::split(items, copy, boost::is_any_of(" \t"), boost::token_compress_on);
}

void readWeightsFile(ifstream &TRAININ, vector<float> &weights) {
  string line;
  while (getline(TRAININ, line) && line != "")
  {
    vector<string> items;
    splitBySpace(line, items);
    if (items.size() != 1)
    {
        cerr << "Error: weights file should have only one weight per line" << endl;
        exit(-1);
    }
    weights.push_back(boost::lexical_cast<float>(items[0]));
  }
}

void readWordsFile(ifstream &TRAININ, vector<string> &word_list)
{
  string line;
  while (getline(TRAININ, line) && line != "")
  {
    vector<string> words;
    splitBySpace(line, words);
    if (words.size() != 1)
    {
        cerr << "Error: vocabulary file must have only one word per line" << endl;
        exit(-1);
    }
    word_list.push_back(words[0]);
  }
}

void readWordsFile(const string &file, vector<string> &word_list)
{
  cerr << "Reading word list from: " << file<< endl;

  ifstream TRAININ;
  TRAININ.open(file.c_str());
  if (! TRAININ)
  {
    cerr << "Error: can't read word list from file " << file<< endl;
    exit(-1);
  }

  readWordsFile(TRAININ, word_list);
  TRAININ.close();
}

void writeWordsFile(const vector<string> &words, ofstream &file)
{
    for (int i=0; i<words.size(); i++)
    {
	file << words[i] << endl;
    }
}

void writeWordsFile(const vector<string> &words, const string &filename)
{
    ofstream OUT;
    OUT.open(filename.c_str());
    if (! OUT)
    {
      cerr << "Error: can't write to file " << filename << endl;
      exit(-1);
    }
    writeWordsFile(words, OUT);
    OUT.close();
}


// Read a data file of unknown size into a flat vector<int>.
// If this takes too much memory, we should create a vector of minibatches.
void readDataFile(const string &filename, int &ngram_size, vector<int> &data, int minibatch_size)
{
  cerr << "Reading minibatches from file " << filename << ": ";

  ifstream DATAIN(filename.c_str());
  if (!DATAIN)
  {
    cerr << "Error: can't read data from file " << filename<< endl;
    exit(-1);
  }

  vector<int> data_vector;

  string line;
  long long int n_lines = 0;
  while (getline(DATAIN, line))
  {
    vector<string> ngram;
    splitBySpace(line, ngram);

    if (ngram_size == 0)
        ngram_size = ngram.size();

    if (ngram.size() != ngram_size)
    {
        cerr << "Error: expected " << ngram_size << " fields in instance, found " << ngram.size() << endl;
	exit(-1);
    }

    for (int i=0;i<ngram_size;i++)
        data.push_back(boost::lexical_cast<int>(ngram[i]));

    n_lines++;
    if (minibatch_size && n_lines % (minibatch_size * 10000) == 0)
      cerr << n_lines/minibatch_size << "...";
  }
  cerr << "done." << endl;
  DATAIN.close();
}


// Read a data file of unknown size into a flat vector<int>.
// If this takes too much memory, we should create a vector of minibatches.
void readSentFile(const string &filename, 
				vector<vector <int> > &data, 
				int minibatch_size,
				data_size_t &num_tokens)
{
  cerr << "Reading input sentences from file " << filename << ": ";

  ifstream DATAIN(filename.c_str());
  if (!DATAIN)
  {
    cerr << "Error: can't read data from file " << filename<< endl;
    exit(-1);
  }

  vector<int> data_vector;

  string line;
  long long int n_lines = 0;
  while (getline(DATAIN, line))
  {
    vector<string> ngram;
    splitBySpace(line, ngram);
	
	/*
    if (ngram_size == 0)
        ngram_size = ngram.size();

    if (ngram.size() != ngram_size)
    {
        cerr << "Error: expected " << ngram_size << " fields in instance, found " << ngram.size() << endl;
	exit(-1);
    }
	*/
	vector<int> int_ngram;
    for (int i=0;i<ngram.size();i++)
        int_ngram.push_back(boost::lexical_cast<int>(ngram[i]));

	data.push_back(int_ngram);
	num_tokens += int_ngram.size();
	
    n_lines++;
    if (minibatch_size && n_lines % (minibatch_size * 10000) == 0)
      cerr << n_lines/minibatch_size << "...";
  }
  cerr << "done." << endl;
  DATAIN.close();
}

void miniBatchify(const std::vector<std::vector <int> > &sentences, 
				 std::vector<int > &minibatch_sentences,
				const int minibatch_start_index,
				const int minibatch_end_index,
				unsigned int &max_sent_len,
				bool is_input,
				unsigned int &minibatch_tokens){
	//cerr<<"minibatch start index is "<<minibatch_start_index<<endl;
	//cerr<<"minibatch end index is "<<minibatch_end_index<<endl;
	//First go over all the sentences and get the longest sentence
	max_sent_len = 0;
	//cerr<<"max sent len boefore is "<<max_sent_len<<endl;
	for (int index=minibatch_start_index; index<= minibatch_end_index; index++){
		//cerr<<"sent len "<<sentences[index].size()<<endl;
		//cerr<<"max sent len in loop is "<<max_sent_len<<endl;
		//cerr<<max_sent_len < sentences[index].size()<<endl;
		if (max_sent_len < sentences[index].size()) {
			//cerr<<"Ths is true"<<endl;
			max_sent_len = sentences[index].size();
			//cerr<<"max_sent_len is now"<<max_sent_len<<endl;
		}
	}
	//Now createing the vector of vectors which is the minibatch size
	//Note that I could do this already with the training data. 
	//for ()
	//cerr<<"max sent len is "<<max_sent_len<<endl;
	for (int index=minibatch_start_index; index<= minibatch_end_index; index++){
		//vector<int> extended_sent(max_sent_len,-1);
		int sent_index=0;
		for (;sent_index<sentences[index].size(); sent_index++){
			minibatch_sentences.push_back(sentences[index][sent_index]);
			minibatch_tokens++;
		}
		//Now padding the rest with -1
		for (;sent_index<max_sent_len; sent_index++){
			//If its the output sentence, then set the output label to -1
			minibatch_sentences.push_back((is_input)? 0:-1);
		}
	}
}

double logadd(double x, double y)
{
    if (x > y)
        return x + log1p(std::exp(y-x));
    else
        return y + log1p(std::exp(x-y));
}

#ifdef USE_CHRONO
void Timer::start(int i)
{
    m_start[i] = clock_type::now();
}

void Timer::stop(int i)
{
    m_total[i] += clock_type::now() - m_start[i];
}

void Timer::reset(int i) { m_total[i] = duration_type(); }

double Timer::get(int i) const
{
    return boost::chrono::duration<double>(m_total[i]).count();
}

Timer timer(20);
#endif

int setup_threads(int n_threads)
{
    #ifdef _OPENMP
    if (n_threads)
        omp_set_num_threads(n_threads);
    n_threads = omp_get_max_threads();
    if (n_threads > 1)
        cerr << "Using " << n_threads << " threads" << endl;

    Eigen::initParallel();
    Eigen::setNbThreads(n_threads);

    #ifdef MKL_SINGLE
    // Set the threading layer to match the compiler.
    // This lets MKL automatically go single-threaded in parallel regions.
    #ifdef __INTEL_COMPILER
    mkl_set_threading_layer(MKL_THREADING_INTEL);
    #elif defined __GNUC__
    mkl_set_threading_layer(MKL_THREADING_GNU);
    #endif
    mkl_set_num_threads(n_threads);
    #endif
    #endif

    return n_threads;
}

} // namespace nplm
