#ifndef TYPES_HPP
#define TYPES_HPP

#include <cmath>
#include <string>
#include <vector>
#include <boost/cstdint.hpp>
#include <limits>

namespace biglm{

typedef double weight_type;
const weight_type IMPOSSIBLE = -HUGE_VAL;

typedef unsigned long block_type;
const size_t bits_per_block = (std::numeric_limits<block_type>::digits);
  //typedef std::size_t size_type;
typedef boost::uint64_t size_type;
typedef unsigned char byte_type;

template<typename T>
struct bytes {
  static const byte_type *data(const T& key) { return reinterpret_cast<const byte_type *>(&key); }
  static size_type size(const T& key) { return sizeof(T); }
};

template<>
struct bytes<std::string> {
  static const byte_type *data(const std::string& key) { return reinterpret_cast<const byte_type *>(key.data()); }
  static size_type size(const std::string& key) { return key.size(); }
};

template<typename U>
struct bytes<std::vector<U> > {
  static const byte_type *data(const std::vector<U>& key) { return reinterpret_cast<const byte_type *>(&key[0]); }
  static size_type size(const std::vector<U>& key) { return key.size() * sizeof(U); }
};

} //namespace nplm

#endif
