#ifndef KUU_GRAPH_HPP
#define KUU_GRAPH_HPP

#include "config.hpp"
#include "mixin/non_copyable.hpp"
#include "mixin/non_movable.hpp"
#include "tensor.hpp"
#include <memory>
#include <unordered_map>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xexpression.hpp>

namespace kuu {
class traceable_function;
class module;
class graph;
class optimizer;
template <typename T> class tensor_container;
namespace detail {
extern std::shared_ptr<graph> g;
} // namespace detail

namespace trace {
template <typename Function>
void register_node(std::initializer_list<tensor> inputs, tensor &output);
void run_backward(const tensor &root);
} // namespace trace

class graph : private non_copyable<graph>, private non_movable<graph> {
  template <typename Function>
  friend void trace::register_node(std::initializer_list<tensor> inputs,
                                   tensor &output);
  friend void trace::run_backward(const tensor &root);
  friend class optimizer;

public:
  graph() = default;
  ~graph() = default;

  void show_nodes() const;

  std::string node_name(const id_type node_id);

private:
  std::unordered_map<id_type, std::shared_ptr<traceable_function>> nodes_;
  std::unordered_map<id_type, std::vector<tensor>> operator_inputs_;
  std::unordered_map<id_type, std::vector<tensor>> backward_stack_;

  void clear() {
    nodes_.clear();
    operator_inputs_.clear();
    backward_stack_.clear();
  }
};

namespace trace {
template <typename Function>
void register_node(std::initializer_list<tensor> inputs, tensor &output) {
  auto node = std::make_unique<Function>();
  node->backward_function = Function::backward;
  id_type id = node->id();
  detail::g->nodes_[id] = std::move(node);
  output.set_creator_id(id);
  detail::g->operator_inputs_[id] = std::move(std::vector<tensor>{inputs});
}
} // namespace trace

} // namespace kuu

#endif // KUU_GRAPH_HPP
