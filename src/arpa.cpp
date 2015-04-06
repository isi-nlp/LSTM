#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cstdio>

#include <boost/algorithm/string_regex.hpp>
#include <boost/lexical_cast.hpp>

#include "arpa.hpp"
#include "types.hpp"

using namespace std;
using namespace boost;
using namespace boost::algorithm;

namespace biglm{

regex gram_header("\\\\(\\d+)-grams:");
regex data_line("ngram (\\d+)=(\\d+)");
regex whitespace("\\s+");

arpa_reader::arpa_reader(istream &file, bool convert_numbers) : m_file(file), m_convert_numbers(convert_numbers) {
  string line;
  smatch what;
  int order;
  m_state = 0;

  if (sizeof(size_type) > sizeof(streamoff))
    cerr << "warning: streamoff has fewer bits than addresses do\n";

  while (!getline(m_file, line).eof()) {
    trim_right(line);
    if (line == "")
      continue;
    else if (line[0] == '\\') {
      if (line == "\\data\\") {
	m_state = 1;
      } else if (regex_match(line, what, gram_header)) {
	order = lexical_cast<int>(what[1].str());
	m_state = 2;
	m_order = order;
	break;
      } else {
	throw runtime_error("unexpected header");
      }
    } else if (m_state == 1 && regex_match(line, what, data_line)) {
      order = lexical_cast<int>(what[1].str());
      if (order > m_counts.size())
	m_counts.resize(order);
      m_counts[order-1] = lexical_cast<int>(what[2].str());
    }
  }
}

int arpa_reader::n_orders() {
  if (m_state == 0 || m_state == 1)
    throw logic_error("count requested before available");
  return m_counts.size();
}

int arpa_reader::n_ngrams(int order) {
  if (m_state == 0 || m_state == 1)
    throw logic_error("count requested before available");
  return m_counts[order-1];
}

int arpa_reader::next_order() {
  string line;
  smatch what;
  if (m_state == 2) {
    m_state = 3;
  } else if (m_state == 3) {
    while (!getline(m_file, line).eof()) {
      trim_right(line);
      if (line != "")
	break;
    }
    if (m_file.eof())
      throw runtime_error("unexpected end of file");
    else if (line == "\\end\\")
      return 0;
    else if (regex_match(line, what, gram_header))
      m_order = lexical_cast<int>(what[1].str());
    else
      throw runtime_error("section not completely read");
  } 
  m_pos = m_file.tellg();
  return m_order;
}

void arpa_reader::rewind_ngrams() {
  if (m_state != 3)
    throw runtime_error("rewind requested at wrong time");
  m_file.seekg(m_pos);
}

arpa_line arpa_reader::next_ngram() {
  string line, dummy;
  if (m_state != 3)
    throw runtime_error("ngram requested at wrong time");
  while (!getline(m_file, line).eof()) {
    trim_right(line);
    if (line == "") {
      continue;
    } else if (line[0] == '\\') {
      throw runtime_error("unexpected end of section");
    } else {
      arpa_line l(m_order);
      istringstream tokenizer(line);
      string word;

      if (m_convert_numbers)
	tokenizer >> l.prob;
      else
	tokenizer >> dummy;
      for (int k=0; k<m_order; k++)
	tokenizer >> l.words[k];

      if (m_convert_numbers && tokenizer >> l.bow)
	l.has_bow = 1;
      else
	l.has_bow = 0;
      return l;
    }
  }
  throw runtime_error("unexpected end of file");
}

}//namespace nplm

