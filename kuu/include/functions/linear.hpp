#ifndef KUU_FUNCTIONS_LINEAR_HPP
#define KUU_FUNCTIONS_LINEAR_HPP

#include "function.hpp"
#include <fstream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xcsv.hpp>

namespace kuu {
namespace function {
struct linear : public virtual traceable_function {
  using self_type = function::linear;

  linear() : traceable_function{1} { set_name("function-linear"); }

  static tensor forward(const tensor &input, const tensor &weight,
                        const tensor &bias = tensor{}) {
    assert(!input.is_empty());
    assert(!weight.is_empty());
    // assert(input.shape().size() == 2);
    assert(weight.shape().size() == 2);
    assert(weight.shape()[0] == input.size() / input.shape()[0]);
    auto x = input.cdata();
    if (2 < x.dimension()) {
      x.reshape({(int)input.shape()[0], -1}); // n, in
    }
    xt::xtensor<value_type, 2> W = weight.cdata(); // in, out

    auto y = xt::linalg::dot(x, W); // n, out

    if (!bias.is_empty()) {
      assert(bias.shape()[0] == weight.shape()[1]);
      auto b = bias.cdata();
      b.reshape({1, bias.shape()[0]}); // {out} to {1, out}
      y += b;
    }
    tensor output{std::move(y), util::requires_grad(input, weight, bias)};
    assert(output.shape()[0] == input.shape()[0]);
    assert(output.shape()[1] == weight.shape()[1]);

    trace::register_node<self_type>({input, weight, bias}, output);

    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    assert(outputs.size() == 1);
    assert(inputs.size() == 3);

    tensor &input = inputs[0];
    tensor &weight = inputs[1];

    auto &x = input.data(); // n, in
    auto x_shape = x.shape();
    const auto &W = weight.data(); // in, out
    auto gy = outputs[0].cgrad();  // n, out

    // db
    if (!inputs[2].is_empty()) {
      tensor &bias = inputs[2];
      auto gb = xt::sum(gy, {0}); // out
      inputs[2].set_grad(std::move(gb));
    }

    // dW
    if (2 < x_shape.size()) {
      x.reshape({(int)x_shape[0], -1});
    }

    auto gW = xt::linalg::dot(xt::transpose(x),
                              gy); // in, out
    inputs[1].set_grad(std::move(gW));

    // dx
    auto gx = xt::linalg::dot(gy, xt::transpose(W)); // n, in
    if (2 < x_shape.size()) {

      gx.reshape(x_shape);
      x.reshape(x_shape);
    }
    inputs[0].set_grad(std::move(gx));
  }
};

} // namespace function
} // namespace kuu

#endif // KUU_FUNCTIONS_LINEAR_HPP