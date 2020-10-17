#ifndef KUU_FUNCTIONS_SOFTMAX_CROSS_ENTROPY_HPP
#define KUU_FUNCTIONS_SOFTMAX_CROSS_ENTROPY_HPP

#include "function.hpp"
#include "math.hpp"
#include "util/converter.hpp"
#include <cassert>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xbroadcast.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

namespace kuu {
namespace function {
class softmax_cross_entropy : virtual public traceable_function {
public:
  enum reduction_type { kMean = 0, kSum = 1 };
  softmax_cross_entropy() : traceable_function{1} {
    set_name("softmax_cross_entropy");
  }
  static tensor forward(const tensor &x, const tensor &t,
                        reduction_type reduction = reduction_type::kMean) {
    assert(2 == x.dim());                 // {N, n_label}
    assert(t.dim() == 1 || t.dim() == 2); // {N, n_label}
    assert(x.shape()[0] == t.shape()[0]);

    xt::xarray<value_type> scores = math::log_softmax(x.cdata(), 1);

    assert(scores.dimension() == 2);
    assert(scores.shape()[1] == x.shape()[1]);

    xt::xtensor<value_type, 0> y;

    if (t.dim() == 1) {
      using index_type = std::array<std::size_t, 2>;
      std::vector<index_type> indices{scores.shape()[0]};
      for (std::size_t i = 0; i < scores.shape()[0]; i++) {
        indices[i] = {i, static_cast<std::size_t>(t.cdata()(i))};
      }
      auto prob = xt::index_view(scores, indices); // {N}

      y = -xt::sum(prob); // sum of ce
      if (reduction == reduction_type::kMean) {
        y /= prob.shape()[0];
      }
    } else {
      xt::xtensor<value_type, 2> target;
      target = t.cdata();

      auto aaa = scores * target;

      auto ce = -xt::sum(aaa, {1}); // {N}
      if (reduction == reduction_type::kMean) {
        y = xt::mean(ce);
      } else if (reduction == reduction_type::kSum) {
        y = xt::sum(ce);
      }
    }

    tensor output{std::move(y), util::requires_grad(x, t)};
    trace::register_node<softmax_cross_entropy>({x, t}, output);
    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    if (!inputs[0].requires_grad()) {
      return;
    }

    auto &x = inputs[0].data();
    auto &gx = inputs[0].grad();
    auto &t = inputs[1].data();

    auto gy = outputs[0].cgrad();

    auto scores = math::log_softmax(x); // {N, n_label}
    scores = xt::exp(scores);

    if (t.dimension() == 2) {
      gx = gy * (scores - t);
    } else if (x.dimension() == 2) {
      gx = scores;
      for (std::size_t i = 0; i < x.shape()[0]; i++) {
        gx(i, t[i]) -= 1;
      }
      gx *= gy;
    } else {
      std::runtime_error("error!");
    }
    gx /= x.shape()[0];
  }
};
} // namespace function
} // namespace kuu
#endif // KUU_FUNCTIONS_SOFTMAX_CROSS_ENTROPY_HPP
