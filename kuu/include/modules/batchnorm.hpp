#ifndef KUU_MODULES_BATCHNORM_HPP
#define KUU_MODULES_BATCHNORM_HPP

#include "functions/batchnorm.hpp"
#include "module.hpp"

namespace kuu {
struct batchnorm_options {
  std::size_t in_channels;
  double eps = 1e-5;
  double momentum = 0.1;
  bool affine = false;
  bool track_running_status = true;
};

class batchnorm_impl : public virtual module {
public:
  batchnorm_impl() = default;
  batchnorm_impl(std::size_t in_channels, double eps = 1e-5, bool affine = true,
                 bool track_running_status = true, double momentum = 0.1);
  batchnorm_impl(const batchnorm_options &options);

  tensor forward(const tensor &input);
  // void pretty_print(std::ostream &stream) const override;

private:
  void reset();
  batchnorm_options options_;
  tensor weight_;
  tensor bias_;
  tensor running_mean_;
  tensor running_var_;
};

batchnorm_impl::batchnorm_impl(std::size_t in_channels, double eps, bool affine,
                               bool track_running_status, double momentum)
    : options_{in_channels, eps, momentum, affine, track_running_status} {
  reset();
}

batchnorm_impl::batchnorm_impl(const batchnorm_options &options)
    : options_{options} {
  reset();
}

void batchnorm_impl::reset() {
  if (options_.affine) {
    weight_ = this->register_parameter(
        "weight", tensor{xt::ones<value_type>({options_.in_channels})}, true);
    // std::cout << "weight: " << shape2string(weight_.shape()) << std::endl;
    bias_ = this->register_parameter(
        "bias", tensor{xt::zeros<value_type>({options_.in_channels})}, true);
    // std::cout << "bias: " << shape2string(bias_.shape()) << std::endl;
    assert(weight_.dim() == 1);
    assert(bias_.dim() == 1);
    assert(weight_.shape()[0] == options_.in_channels);
    assert(bias_.shape()[0] == options_.in_channels);
  }
  if (options_.track_running_status) {
    if (this->is_training()) {
      running_mean_ =
          tensor{xt::zeros<value_type>({options_.in_channels}), false};
      running_var_ =
          tensor{xt::ones<value_type>({options_.in_channels}), false};
      assert(running_mean_.dim() == 1);
      assert(running_var_.dim() == 1);
      assert(running_mean_.shape()[0] == options_.in_channels);
      assert(running_var_.shape()[0] == options_.in_channels);
    } else {
      std::cerr << "In Batch Norm module: track_running_status is set true but "
                   "now is train mode."
                   "Change track_running_status to false."
                << std::endl;
      options_.track_running_status = false;
    }
  }
  this->is_initialized_ = true; // skip initializing parameters
}

tensor batchnorm_impl::forward(const tensor &input) {
  return function::batchnorm::forward(
      input, weight_, bias_, running_mean_, running_var_, options_.eps,
      options_.momentum, this->is_training() && options_.track_running_status);
}

/*
void batchnorm_impl::pretty_print(std::ostream &stream) const override {
  stream << "kuu::batchnorm(in_channels=" << options_.in_channels
         << ", eps=" << options_.eps << ", momentum=" << options_.momentum
         << ", affine=" << std::boolalpha << options_.affine;
  stream << ", track_running_status=" << std::boolalpha
         << options_.track_running_status << ")" << std::endl;
} */

using batchnorm = module_holder<batchnorm_impl>;

} // namespace kuu
#endif // KUU_MODULES_BATCHNORM_HPP