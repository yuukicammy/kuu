#ifndef KUU_OPTIMIZER_HPP
#define KUU_OPTIMIZER_HPP

#include "graph.hpp"
#include "tensor.hpp"
#include <execution>
#include <vector>

namespace kuu {
namespace detail {
extern std::shared_ptr<graph> g;
} // namespace detail

class optimizer {
public:
  optimizer() = default;
  inline optimizer(std::vector<tensor> &&parameters)
      : steps_{0}, parameters_{std::move(parameters)} {}

  void update();

  virtual void apply(tensor &) = 0;

  void clear_grad();
  auto steps() { return steps_; }

protected:
  unsigned long long int steps_;
  std::vector<tensor> parameters_;
};

void optimizer::clear_grad() {
  std::for_each(std::begin(parameters_), std::end(parameters_),
                [](auto &param) { param.clear_grad(); });
}

void optimizer::update() {
  this->steps_++;

  std::for_each(std::begin(parameters_), std::end(parameters_),
                [this](auto &param) { apply(param); });
  detail::g->clear();
}

} // namespace kuu
#endif // KUU_OPTIMIZER_HPP