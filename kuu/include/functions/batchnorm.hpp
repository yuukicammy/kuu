#ifndef KUU_FUNCTIONS_BATCH_NORM_HPP
#define KUU_FUNCTIONS_BATCH_NORM_HPP

#include "function.hpp"
#include <cmath>
#include <execution>
#include <xtensor/xtensor.hpp>

// references:
// https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

namespace kuu {
namespace function {

class batchnorm_1d : virtual public traceable_function {
public:
  batchnorm_1d() : traceable_function{1} { set_name("batchnorm_1d"); }
  static tensor forward(const tensor &data, const tensor &weight,
                        const tensor &bias, const tensor &running_mean,
                        const tensor &running_var, value_type eps = 1e-5,
                        value_type momentum = 0.1,
                        bool track_running_stats = false) {
    auto x = data.cdata();
    auto x_shape = x.shape();
    std::size_t batch_size = x_shape[0];
    std::size_t channels = x_shape[1];

    assert(x.dimension() == 2);
    assert(running_mean.dim() == 1);
    assert(running_var.dim() == 1);
    assert(running_mean.shape()[0] == channels);
    assert(running_var.shape()[0] == channels);

    xt::xtensor<value_type, 2> y;
    if (track_running_stats) {
      auto batch_mean = xt::mean(x, {0});
      auto batch_var = xt::variance(x, {0});

      y = (x - batch_mean) / xt::sqrt(batch_var + eps);
    } else {
      y = (x - running_mean.cdata()) / xt::sqrt(running_var.cdata() + eps);
    }

    // affine
    if (!weight.is_empty()) {
      auto gamma = weight.cdata();
      assert(gamma.dimension() == 1);
      assert(x.shape()[1] == gamma.shape()[0]);
      y *= gamma;
    }
    if (!bias.is_empty()) {
      auto beta = bias.cdata();
      assert(beta.dimension() == 1);
      assert(x.shape()[1] == beta.shape()[0]);
      y += beta;
    }

    tensor output{std::move(y), util::requires_grad(data, weight, bias)};
    if (output.requires_grad()) {
      trace::register_node<batchnorm_1d>(
          {data, weight, bias, running_mean, running_var, tensor{eps},
           tensor{momentum},
           tensor{static_cast<value_type>(track_running_stats)}},
          output);
    }
    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    assert(inputs.size() == 8);
    assert(outputs.size() == 1);

    auto &&gy = outputs[0].cgrad();
    auto &x = inputs[0].data();
    auto &gx = inputs[0].grad();
    auto &running_mean = inputs[3].data();
    auto &running_var = inputs[4].data();
    value_type eps = inputs[5].data()();
    value_type momentum = inputs[6].data()();
    bool track_running_stats = 0 < inputs[7].data()() ? true : false;

    assert(gy.dimension() == x.dimension());
    assert(gy.dimension() == 2);
    assert(gy.size() == x.size());

    std::size_t batch_size = x.shape()[0];

    auto &&batch_mean = xt::mean(x, {0});
    auto &&batch_var = xt::variance(x, {0});

    xt::xtensor<value_type, 2> xhat;
    if (track_running_stats) {
      xhat = (x - batch_mean) / xt::sqrt(batch_var + eps);
    } else {
      xhat = (x - running_mean) / xt::sqrt(running_var + eps);
    }

    if (!inputs[2].is_empty() && inputs[2].requires_grad()) {
      // gradient of beta
      auto gbeta = xt::sum(gy, {0});
      inputs[2].set_grad(gbeta);
    }

    // gradient of gamma
    if (!inputs[1].is_empty() && inputs[1].requires_grad()) {
      auto ggamma = xt::sum(gy * xhat, {0});
      inputs[1].set_grad(ggamma);
    }

    // gradinent of x
    if (inputs[0].requires_grad()) {
      if (track_running_stats) {
        auto gxhat = xt::xarray<value_type>::from_shape(gy.shape());
        if (inputs[1].is_empty()) {
          gxhat = gy;
        } else {
          auto &gamma = inputs[1].data();
          gxhat = gy * gamma;
        }
        auto ivar = 1. / xt::sqrt(batch_var + eps);
        auto givar = xt::sum(gxhat * batch_mean, {0});
        auto gbatch_mean_1 = gxhat * ivar;
        auto gsqrtvar = -1. / batch_var * givar;
        auto gbatch_var = 0.5 * ivar * gsqrtvar;
        auto gsq = 1. / batch_size *
                   xt::ones<value_type>({batch_size, x.shape()[1]}) *
                   gbatch_var;
        auto gbatch_mean_2 = 2 * batch_mean * gsq;
        auto gx_1 = gbatch_mean_1 + gbatch_mean_2;
        auto gbatch_mean = -1 * xt::sum(gbatch_mean_1 + gbatch_mean_2, {0});
        auto gx_2 = 1. / batch_size *
                    xt::ones<value_type>({batch_size, gy.shape()[1]}) *
                    gbatch_mean;
        gx = gx_1 + gx_2;
      } else {
        gx = gy / xt::sqrt(running_var + eps);
      }
    }
    if (track_running_stats) {
      running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
      running_var = momentum * running_var + (1 - momentum) * batch_var;
    }
  }
};

class batchnorm_nd : virtual public traceable_function {
public:
  batchnorm_nd() : traceable_function{1} { set_name("batchnorm_nd"); }

  static tensor forward(const tensor &data, const tensor &weight,
                        const tensor &bias, const tensor &running_mean,
                        const tensor &running_var, value_type eps = 1e-5,
                        value_type momentum = 0.1,
                        bool track_running_stats = true) {

    auto &&x = data.cdata();
    auto x_shape = x.shape();
    std::size_t channels = x_shape[1];

    assert(2 < x.dimension());
    assert(running_mean.dim() == 1);
    assert(running_var.dim() == 1);

    x.reshape({static_cast<int>(x_shape[0]), static_cast<int>(x_shape[1]), -1});
    auto y = xt::xarray<value_type>::from_shape(x.shape());

    assert(running_mean.shape()[0] == x.shape()[1]);
    assert(running_var.shape()[0] == x.shape()[1]);

    if (track_running_stats) {
      xt::xarray<value_type> batch_mean = xt::mean(x, {0, 2});
      xt::xarray<value_type> batch_var = xt::variance(x, {0, 2});
      batch_mean.reshape({1, channels, 1});
      batch_var.reshape({1, channels, 1});

      y = (x - batch_mean) / xt::sqrt(batch_var + eps);
    } else {
      xt::xarray<value_type> mean = running_mean.cdata();
      xt::xarray<value_type> var = running_var.cdata();
      mean.reshape({1, channels, 1});
      var.reshape({1, channels, 1});
      y = (x - mean) / xt::sqrt(var + eps);
    }

    // affine
    if (!weight.is_empty()) {
      auto &&gamma = weight.cdata();
      assert(gamma.dimension() == 1);
      assert(x_shape[1] == gamma.shape()[0]);
      gamma.reshape({1, channels, 1});
      y *= gamma;
    }
    if (!bias.is_empty()) {
      auto &&beta = bias.cdata();
      assert(beta.dimension() == 1);
      assert(x_shape[1] == beta.shape()[0]);
      beta.reshape({1, channels, 1});
      y += beta;
    }

    y.reshape(x_shape);
    tensor output{std::move(y), util::requires_grad(data, weight, bias)};
    if (output.requires_grad()) {
      trace::register_node<batchnorm_nd>(
          {data, weight, bias, running_mean, running_var, tensor{eps},
           tensor{momentum},
           tensor{static_cast<value_type>(track_running_stats)}},
          output);
    }
    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {

    assert(inputs.size() == 8);
    assert(outputs.size() == 1);

    auto gy = outputs[0].cgrad();
    xt::xarray<value_type> &x = inputs[0].data();
    auto x_shape = x.shape();
    auto &gx = inputs[0].grad();
    auto &running_mean = inputs[3].data();
    auto &running_var = inputs[4].data();
    value_type eps = inputs[5].data()();
    value_type momentum = inputs[6].data()();
    bool track_running_stats = 0 < inputs[7].data()() ? true : false;

    assert(gy.dimension() == x.dimension());
    assert(2 < gy.dimension());
    assert(gy.size() == x.size());

    std::size_t batch_size = x_shape[0];
    std::size_t channels = x_shape[1];

    gy.reshape(
        {static_cast<int>(x.shape()[0]), static_cast<int>(x.shape()[1]), -1});
    x.reshape(gy.shape());

    xt::xarray<value_type> batch_mean = xt::mean(x, {0, 2});
    xt::xarray<value_type> batch_var = xt::variance(x, {0, 2});
    batch_mean.reshape({1, channels, 1});
    batch_var.reshape({1, channels, 1});

    auto xhat = xt::xtensor<value_type, 3>::from_shape(x.shape());
    if (track_running_stats) {
      xhat = (x - batch_mean) / xt::sqrt(batch_var + eps);
    } else {
      running_mean.reshape({1, channels, 1});
      running_var.reshape({1, channels, 1});
      xhat = (x - running_mean) / xt::sqrt(running_var + eps);
    }

    // gradient of beta
    if (!inputs[2].is_empty() && inputs[2].requires_grad()) {
      auto gbeta = xt::sum(gy, {0, 2});
      inputs[2].set_grad(std::move(gbeta));
    }

    // gradient of gamma
    if (!inputs[1].is_empty() && inputs[1].requires_grad()) {
      auto ggamma = xt::sum(gy * xhat, {0, 2});
      inputs[1].set_grad(std::move(ggamma));
    }

    // gradinent of x
    if (inputs[0].requires_grad()) {
      if (track_running_stats) {
        batch_var.reshape({channels, 1});
        xt::xtensor<value_type, 3> gxhat{xhat.shape()};
        if (inputs[1].is_empty()) {
          gxhat = gy;
        } else {
          xt::xarray<value_type> &&gamma = inputs[1].cdata();
          gamma.reshape({1, channels, 1});
          gxhat = gy * gamma;
        }

        auto ivar = 1. / xt::sqrt(batch_var + eps);
        auto givar = xt::sum(gxhat * batch_mean, {0});
        auto gbatch_mean_1 = gxhat * ivar;
        auto gsqrtvar = -1. / batch_var * givar;
        auto gbatch_var = 0.5 * ivar * gsqrtvar;
        auto gsq = 1. / batch_size * xt::ones_like(gy) * gbatch_var;
        auto gbatch_mean_2 = 2 * batch_mean * gsq;
        xt::xarray<value_type> gx_1 = gbatch_mean_1 + gbatch_mean_2;
        auto gbatch_mean = -1 * xt::sum(gbatch_mean_1 + gbatch_mean_2, {0});
        xt::xarray<value_type> gx_2 =
            1. / batch_size * xt::ones_like(gy) * gbatch_mean;

        gx_1.reshape(x_shape);
        gx_2.reshape(x_shape);
        x.reshape(x_shape);
        running_mean.reshape({channels});
        running_var.reshape({channels});

        gx = gx_1 + gx_2;
      } else {
        gx = gy / xt::sqrt(running_var + eps);
      }
    }
    if (track_running_stats) {
      batch_mean.reshape({channels});
      batch_var.reshape({channels});
      running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
      running_var = momentum * running_var + (1 - momentum) * batch_var;
    }
  }
};

class batchnorm {
public:
  batchnorm() {}

  static tensor forward(const tensor &data, const tensor &weight,
                        const tensor &bias, const tensor &running_mean,
                        const tensor &running_var, value_type eps = 1e-5,
                        value_type momentum = 0.1,
                        bool track_running_stats = false) {
    if (data.dim() == 2) {
      return batchnorm_1d::forward(data, weight, bias, running_mean,
                                   running_var, eps, momentum,
                                   track_running_stats);
    } else if (2 < data.dim()) {
      return batchnorm_nd::forward(data, weight, bias, running_mean,
                                   running_var, eps, momentum,
                                   track_running_stats);
    }
    throw std::runtime_error("batchnorm arguments have invalid shape.");
  }
};
} // namespace function
} // namespace kuu

#endif // KUU_FUNCTIONS_BATCH_NORM_HPP
