#ifndef KUU_FUNCTIONS_RELU_HPP
#define KUU_FUNCTIONS_RELU_HPP

#include "function.hpp"
#include <cassert>
#include <string>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xtensor.hpp>

namespace kuu {
namespace function {
class relu : public traceable_function {
  using self_type = relu;

public:
  relu() : traceable_function{1} { set_name("activation-relu"); }

  static tensor forward(const tensor &input) {
    auto x = input.cdata();
    auto y = xt::fmax(0, x);

    tensor output{std::move(y), util::requires_grad(input)};
    trace::register_node<self_type>({input}, output);
    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    assert(outputs.size() == 1);
    assert(inputs.size() == 1);
    tensor &input = inputs[0];
    const auto &x = input.data();
    auto dy = outputs[0].cgrad();
    auto dx = dy * (x > 0);
    input.set_grad(dx);
  }
};
} // namespace function
} // namespace kuu

#endif // KUU_FUNCTIONS_RELU_HPP