#ifndef KUU_UTIL_IO_HPP_
#define KUU_UTIL_IO_HPP_

#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace kuu {
namespace io {

template <typename _Ty>
std::ostream &operator<<(std::ostream &ostr, const std::vector<_Ty> &v) {
  if (v.empty()) {
    ostr << "{ }";
    return ostr;
  }
  ostr << "{" << v.front();
  for (auto itr = ++v.begin(); itr != v.end(); itr++) {
    ostr << ", " << *itr;
  }
  ostr << "}";
  return ostr;
}

#endif // KUU_UTIL_IO_HPP_