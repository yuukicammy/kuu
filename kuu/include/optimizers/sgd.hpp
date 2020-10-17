#ifndef KUU_OPTIMIZERS_SGD_HPP
#define KUU_OPTIMIZERS_SGD_HPP

#include "optimizer.hpp"
#include "tensor.hpp"
#include <cassert>

namespace kuu {

class sgd : public optimizer {
public:
  struct options {
    double learning_rate;
    double weight_decay;
    double momentum;
    double dampening;
    bool nesterov;
  };

  sgd(std::vector<tensor> &&parameters, options &&hyperparams)
      : optimizer{std::move(parameters)}, hyperparams_{std::move(hyperparams)} {
    assert(0 < hyperparams_.learning_rate);
    assert(0 <= hyperparams_.weight_decay);
    if (hyperparams_.nesterov && 0 < hyperparams_.momentum) {
      assert(hyperparams_.dampening == 0);
    }
  }

  void apply(tensor &parameter) override;

private:
  options hyperparams_;
  std::unordered_map<id_type, xt::xarray<float>> velocity_;
};

void sgd::apply(tensor &parameter) {
  if (!parameter.requires_grad()) {
    return;
  }
  auto grad = parameter.cgrad();
  auto &data = parameter.data();
  if (hyperparams_.weight_decay != 0) {
    grad += (hyperparams_.weight_decay * data);
  }

  const id_type id = parameter.id();
  if (0 != hyperparams_.momentum) {
    if (velocity_.find(id) == velocity_.end()) {
      velocity_[id] = grad;
    } else {
      velocity_[id] = hyperparams_.momentum * velocity_[id] +
                      (1 - hyperparams_.dampening) * grad;
    }
    if (hyperparams_.nesterov) {
      grad += velocity_[id] * hyperparams_.momentum;
    } else {
      grad = velocity_[id];
    }
  }
  data -= hyperparams_.learning_rate * grad;
}

} // namespace kuu

#endif // KUU_OPTIMIZERS_SGD_HPP