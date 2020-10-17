#ifndef KUU_INITIALIZER_HPP
#define KUU_INITIALIZER_HPP

#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xshape.hpp>

namespace kuu {

inline void zeros(tensor &target) {
  target = xt::zeros<value_type>(target.shape());
  target.set_grad(xt::zeros_like(target.data()));
}

inline void ones(tensor &target) {
  target = xt::ones<value_type>(target.shape());
  target.set_grad(xt::zeros_like(target.data()));
}

template <typename D = value_type>
inline void constant(tensor &target, const D val) {
  target = xt::ones<value_type>(target.shape()) * val;
  target.set_grad(xt::zeros_like(target.data()));
}

template <typename D = value_type>
inline void uniform(tensor &target, const D lower = 0, const D upper = 1) {
  target = xt::random::rand<value_type>(target.shape(), lower, upper);
  target.set_grad(xt::zeros_like(target.data()));
}

template <typename D = value_type>
inline void normal(tensor &target, const D mean = 0, const D std_dev = 1) {
  target = xt::random::randn<value_type>(target.shape(), mean, std_dev);
  target.set_grad(xt::zeros_like(target.data()));
}

template <typename D = value_type>
inline void he_normal(tensor &target) {
  float n;

  if (target.dim() == 1) {
    n = static_cast<float>(target.shape()[0]);
  } else if (target.dim() == 2) {
    n = static_cast<float>(target.shape()[0] * target.shape()[1]);
  } else if (target.dim() == 4) {
    n = static_cast<float>(target.shape()[0] * target.shape()[2] * target.shape()[3]);
  } else {
    std::string s_dim = boost::lexical_cast<std::string>(target.dim());
    throw std::runtime_error("input tensor has unexpected dimension:" + s_dim);
  }

  target = xt::random::randn<value_type>(target.shape(), 0.0, sqrt(2.0 / n));
  target.set_grad(xt::zeros_like(target.data()));
}


namespace initializer {
static std::function<void(tensor &)> zeros = kuu::zeros;
static std::function<void(tensor &)> ones = kuu::ones;
static std::function<void(tensor &, value_type)> constant = kuu::constant<>;
static std::function<void(tensor &, value_type, value_type)> uniform =
    kuu::uniform<>;
static std::function<void(tensor &, value_type, value_type)> normal =
    kuu::normal<>;
static std::function<void(tensor &)> he_normal =
    kuu::he_normal<>;

} // namespace initializer

} // namespace kuu
#endif // KUU_INITIALIZER_HPP
