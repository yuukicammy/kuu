#ifndef KUU_MODULES_LINEAR_HPP
#define KUU_MODULES_LINEAR_HPP

#include "functions/linear.hpp"
#include "module.hpp"
#include "tensor.hpp"
#include "util/converter.hpp"
#include "util/io.hpp"
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xexpression.hpp>

namespace kuu {

struct linear_options {
  size_t in_size;
  size_t out_size;
  bool use_bias;
};

class linear_impl : public virtual module {
public:
  using self_type = linear_impl;
  linear_impl() = default;
  linear_impl(linear_options &&options);

  ~linear_impl() = default;

  tensor forward(const tensor &in);

private:
  linear_options options_;
  tensor weight_, bias_;
};

linear_impl::linear_impl(linear_options &&options)
    : options_{std::move(options)},
      weight_{register_parameter(
          "linear-weight", tensor{{options.in_size, options.out_size}, true})} {
  if (options_.use_bias) {
    bias_ = register_parameter(
        "linear-bias",
        tensor{std::vector<std::size_t>{options_.out_size}, true});
    assert(bias_.id() == params_["linear-bias"].id());
  }
  assert(weight_.id() == params_["linear-weight"].id());
}

tensor linear_impl::forward(const tensor &input) {
  tensor output = options_.use_bias
                      ? function::linear::forward(input, weight_, bias_)
                      : function::linear::forward(input, weight_, tensor{});
  return output;
}

using linear = module_holder<linear_impl>;
} // namespace kuu
#endif // KUU_MODULES_LINEAR_HPP
