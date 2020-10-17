#ifndef KUU_MATH_HPP
#define KUU_MATH_HPP

#include <xtensor/xexpression.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xoperation.hpp>

namespace kuu {
namespace math {
template <class E> auto softmax(const xt::xexpression<E> &e, size_t axis = 1) {
  auto x = e.derived_cast();
  auto xmax = xt::amax(x, {axis});

  for (std::size_t i = 0; i < x.shape()[0]; i++) {
    xt::view(x, i, xt::all()) -= xmax(i);
  }
  return xt::eval(xt::exp(x - xmax) / xt::sum(xt::exp(x - xmax), {axis}));
}

template <class E>
auto log_softmax(const xt::xexpression<E> &e, size_t axis = 1) {
  xt::xarray<value_type> x = e.derived_cast();
  auto y = x;
  auto xmax = xt::amax(x, {axis});
  for (std::size_t i = 0; i < x.shape()[0]; i++) {
    xt::view(y, i, xt::all()) -= xmax(i);
  }
  auto log_sum_exp = xt::log(xt::sum(xt::exp(y), {axis})) + xmax;

  for (std::size_t i = 0; i < x.shape()[0]; i++) {
    xt::view(x, i, xt::all()) -= log_sum_exp(i);
  }

  return x;
}

template <class E> inline auto sigmoid(const xt::xexpression<E> &x) {
  return xt::eval(1. / (1. + xt::exp(-x.derived_cast())));
}
} // namespace math
} // namespace kuu

#endif // KUU_MATH_HPP