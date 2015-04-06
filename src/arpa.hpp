#ifndef ARPA_HPP
#define ARPA_HPP

#include <iostream>
#include <vector>

#include "types.hpp"

namespace biglm{

struct arpa_line {
  std::vector<std::string> words;
  weight_type prob, bow;
  bool has_bow;

  arpa_line() {}
  arpa_line(int order) : words(order) {}
};

class arpa_reader {
  int m_state, m_order;
  std::vector<int> m_counts;
  std::istream &m_file;
  std::istream::pos_type m_pos;
  bool m_convert_numbers;

public:
  arpa_reader(std::istream &file, bool convert_numbers);
  int n_orders();
  int n_ngrams(int order);
  int next_order();
  arpa_line next_ngram();
  void rewind_ngrams();
};

} //namespace nplm

#endif
