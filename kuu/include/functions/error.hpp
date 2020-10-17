#ifndef KUU_FUNCTIONS_ERROR_HPP
#define KUU_FUNCTIONS_ERROR_HPP

#include "function.hpp"
#include "util/converter.hpp"
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

namespace kuu {
namespace function {
class mean_squared_error : virtual public traceable_function {
public:
  mean_squared_error() : traceable_function{1} {
    set_name("mean_squared_error");
  }
  ~mean_squared_error() = default;

  static tensor forward(const tensor &x0, const tensor &x1) {

    auto diff = xt::flatten(x0.cdata()) - xt::flatten(x1.cdata());
    tensor::tensor_type mean;
    mean = xt::mean(xt::square(std::move(diff))); // mean all

    tensor output{std::move(mean), util::requires_grad(x0, x1)};

    trace::register_node<mean_squared_error>({x0, x1}, output);

    return output;
  }

  tensor operator()(const tensor &x0, const tensor &x1) {
    return forward(x0, x1);
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    const auto &x0 = inputs[0].data();
    const auto &x1 = inputs[1].data();
    auto gy = outputs[0].cgrad();

    tensor_type diff = xt::flatten(x0) - xt::flatten(x1);

    xt::xarray<value_type> gx =
        xt::xscalar<value_type>(2. / diff.size()) * gy * diff;

    std::cout << "gx\n" << xt::mean(gx, {0}) << std::endl;

    if (inputs[0].requires_grad()) {
      inputs[0].set_grad(gx.reshape(inputs[0].shape()));
    }
    if (inputs[1].requires_grad()) {
      inputs[1].set_grad(std::move(-gx.reshape(inputs[1].shape())));
    }
  }
};

} // namespace function
} // namespace kuu

#endif // KUU_FUNCTIONS_ERROR_HPP