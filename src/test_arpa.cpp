#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>

//#include "cmph.hpp"
//#include "biglm.hpp"
//#include "quantizer.hpp"
#include "arpa.hpp"
#include "arpaMultinomial.h"

#include <boost/algorithm/string_regex.hpp>
#include <boost/program_options.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/algorithm/string/join.hpp>

//using namespace std;
using namespace boost;
using namespace biglm;
using namespace nplm;
//using boost::random::mersenne_twister_engine;

int main (int argc, char *argv[]) {
  /*
  using namespace boost::program_options;

  options_description optdes("Allowed options");
  optdes.add_options()
    ("help,h", "display help message")
    ("mph-file,m", value<string>(), "Use vocabulary and perfect hash functions from file")
    ("mph-only", "Only generate vocabulary and perfect hash functions")
    ("input-file", value<string>(), "Input language model (ARPA format)")
    ("quantizer-file,q", value<string>(), "File containing interval means")
    ("output-file,o", value<string>(), "Output file (biglm format)")
    ("checksum-bits,k", value<int>()->default_value(8), "bits for checksum (higher is more accurate)")
    ("b", value<int>()->default_value(175), "b parameter to BRZ")
    ("graphsize", value<double>()->default_value(2.9), "graph size parameter to BRZ (>= 2.6)")
    ("debug,d", "debug mode");

  positional_options_description poptdes;
  poptdes.add("input-file", 1);
  poptdes.add("output-file", 1);


  variables_map vm;
  store(command_line_parser(argc, argv).options(optdes).positional(poptdes).run(), vm);
  notify(vm);


  // validate combination of options
  int error = 0;


  if (vm.count("help") || error) {
    cout << "make_biglm <input-file> <output-file>\n\n";
    cout << optdes << "\n";
    return 0;
  }
  */
  /*
  // CREATING an input file stream 
  //ifstream in(vm["input-file"].as<string>().c_str());
  string file = string("e.blanks.lm.3gram.test");
  ifstream in(file.c_str());
  if (in.fail()) {
    cerr << "couldn't open language model\n";
    exit(1);
  }
  
  int order;
  arpa_reader r(in, 1);
  arpa_line l;
  vector<string> vocab;
  size_type vocab_size;

  order = r.n_orders();

  int o;
  while ((o = r.next_order())) {
    cout<<"o is "<<o<<endl;
    if (o == 1) {
      cerr << "building perfect hash for vocab (" << r.n_ngrams(o) << " keys)...";

      for (int i=0; i<r.n_ngrams(o); i++) {
        arpa_line l = r.next_ngram();
        vocab.push_back(l.words[0]);
        cout<<"The line is "<<l.words[0]<<endl;
        cout<<"The prob is "<<l.prob<<" and bow is "<<l.bow<<endl;
        //vocab.push_back(l.words[0]);
      }
    } else {
      for (int i=0; i<r.n_ngrams(o); i++) {
        arpa_line l = r.next_ngram();
        cout<<"The line is ";
        for (int j=0; j<l.words.size();j++) { 
          cout<<l.words[j]<<" ";
        }
        cout<<endl;
        cout<<"The prob is "<<l.prob<<endl;
      }
    }
  }
  */
  std::cout<<"creating the arpa sampler"<<endl;
  //string file = string("e.blanks.lm.3gram.test.integerized");
  std::string file = std::string("e.blanks.lm.3gram.test.unk.integerized");
  unsigned int count = 27;
  Arpa_multinomial<unsigned int> mult(count,file);
  //unsigned seed = std::time(0);
  unsigned int seed = 1234; //for testing only
  boost::random::mt19937 rng(seed);
  std::vector<unsigned int> samples = std::vector<unsigned int>();
  std::vector<double> probs;
  std::vector<unsigned int> context = std::vector<unsigned int>();
  context.push_back(20);
  context.push_back(19);
  mult.sample(rng,
      100,
      context,
      samples,
      probs);
  cerr<<"we will now print the samples "<<endl;
  cerr<<"Size of samples is "<<samples.size()<<endl;
  for (int i=0; i<100; i++){
    cout<<"sample "<<i<<" was "<<samples[i]<<endl;
    cout<<"sample "<<i<<"logprob was "<<probs[i]<<endl;
  }

  return(0);
}
