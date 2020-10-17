#ifndef KUU_MODULES_CONVOLUTION_HPP
#define KUU_MODULES_CONVOLUTION_HPP

#include "functions/convolution.hpp"
#include "module.hpp"

namespace kuu {

template <std::size_t D> struct conv_options {
  std::size_t in_channels;
  std::size_t out_channels;
  exarray<D> kernel_size;
  exarray<D> stride = 1;
  exarray<D> padding = 0;
  exarray<D> dilation = 1;
  bool use_bias = false;
  bool transposed = false;
  exarray<D> out_padding = 0;
};

class conv2d_impl : public virtual module {
public:
  using self_type = conv2d_impl;
  conv2d_impl() = default;
  conv2d_impl(const std::size_t in_channels, const std::size_t out_channels,
              const exarray<2> kernel_size);
  explicit conv2d_impl(conv_options<2> &&options);
  tensor forward(const tensor &input);

private:
  void reset();
  conv_options<2> options_;
  tensor weight_;
  tensor bias_;
};

void conv2d_impl::reset() {
  std::vector<std::size_t> weight_shape = {
      options_.out_channels, options_.in_channels,
      options_.kernel_size.get<0>(), options_.kernel_size.get<1>()};

  weight_ = this->register_parameter("conv2d weight",
                                     tensor{std::move(weight_shape)}, true);
  if (options_.use_bias) {
    bias_ = this->register_parameter(
        "conv2d bias", tensor{std::vector<std::size_t>{options_.out_channels}},
        true);
  }
}

conv2d_impl::conv2d_impl(const std::size_t in_channels,
                         const std::size_t out_channels,
                         const exarray<2> kernel_size)
    : options_{conv_options<2>{in_channels, out_channels, kernel_size}} {
  reset();
}

conv2d_impl::conv2d_impl(conv_options<2> &&options)
    : options_{std::move(options)} {
  reset();
}

tensor conv2d_impl::forward(const tensor &input) {
  return function::convolution_2d::forward(input, weight_, bias_,
                                           options_.stride, options_.padding,
                                           options_.dilation);
}

using conv2d = module_holder<conv2d_impl>;
} // namespace kuu

#endif // KUU_MODULES_CONVOLUTION_HPP