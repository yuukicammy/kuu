#ifndef KUU_UTIL_CONVERTE_HPP
#define KUU_UTIL_CONVERTE_HPP

#include <sstream>
#include <string>
#include <xtensor/xio.hpp>

namespace kuu {

template <class S> std::string shape2string(S shape, std::string del = ", ") {
  std::stringstream ss;
  if (shape.size() == 0) {
    return "{ }";
  } else {
    ss << "{ " << shape[0];
    std::for_each(shape.begin() + 1, shape.end(),
                  [&ss, del](auto s) { ss << del << s; });
    ss << " }";
  }
  return ss.str();
}

template <typename Tensor>
std::string tensor2json(const Tensor &tensor,
                        const bool include_row_data = false) {
  std::stringstream ss;
  std::string n = "\n";
  std::string c = "\"";
  ss << c << "tensor" << c << ": {" << n;
  ss << c << "shape" << c << ": " << shape2string(tensor.shape(), ",") << ","
     << n;
  if (!tensor.is_empty()) {
    ss << c << "data.shape" << c << ": " << shape2string(tensor.cdata().shape())
       << "," << n;
    ss << c << "grad.shape" << c << ": " << shape2string(tensor.cgrad().shape())
       << "," << n;
  } else {
    ss << c << "data.shape" << c << ": "
       << "{}"
       << "," << n;
    ss << c << "grad.shape" << c << ": "
       << "{}"
       << "," << n;
  }

  if (include_row_data) {
    if (!tensor.is_empty()) {
      ss << c << "data" << c << ": " << n << tensor.cdata() << "," << n;
      ss << c << "grad" << c << ": " << n << tensor.cgrad() << "," << n;
    }
  }
  ss << "}" << n;
  return ss.str();
}

// TODO
template <typename Tensor>
std::string vectensor2json(const std::vector<Tensor> &tensors) {
  if (tensors.size() == 0) {
    return "{ }";
  }
  std::string res = "{ \n";
  std::for_each(tensors.begin(), tensors.end(), [&res](auto &t) {
    res += tensor2json(t);
    res += "\n";
  });
  res += "}\n";
  return res;
}
} // namespace kuu

#endif // KUU_UTILS_CONVERTE_HPP