#include "graph.hpp"
#include "function.hpp"
#include "tensor.hpp"
#include "util/util.hpp"

namespace kuu {

namespace detail {
std::shared_ptr<graph> g = std::make_shared<graph>();
} // namespace detail

void graph::show_nodes() const {
  std::cout << "node size: " << detail::g->nodes_.size() << std::endl;
  for_each(std::begin(detail::g->nodes_), std::end(detail::g->nodes_),
           [](auto node) {
             std::cout << node.first << ", " << node.second->id() << std::endl;
           });
}

std::string graph::node_name(const id_type node_id) {
  if (nodes_.find(node_id) == nodes_.end()) {
    return "";
  } else {
    return nodes_[node_id]->name();
  }
}

namespace trace {
void run_backward(const tensor &root) {
  id_type node_id = root.creator_id();
  if (node_id == "") {
    return;
  } else {
    assert(util::find(detail::g->nodes_, node_id));
  }

  bool done_backward = false;

  if (util::find(detail::g->operator_inputs_, node_id) &&
      detail::g->operator_inputs_[node_id].size() > 0) {
    if (detail::g->nodes_[node_id]->n_output() == 1) {
      detail::g->nodes_[node_id]->backward_function(
          {root}, detail::g->operator_inputs_[node_id]);
      done_backward = true;
    } else if (util::find(detail::g->backward_stack_, node_id)) {
      detail::g->backward_stack_[node_id].push_back(root);
      if (detail::g->backward_stack_[node_id].size() ==
          detail::g->nodes_[node_id]->n_output()) {
        std::vector<tensor> outputs{
            std::move(detail::g->backward_stack_[node_id])};
        detail::g->backward_stack_[node_id] = std::vector<tensor>{};
        detail::g->nodes_[node_id]->backward_function(
            outputs, detail::g->operator_inputs_[node_id]);
        done_backward = true;
      } else {
        assert(detail::g->backward_stack_[node_id].size() <
               detail::g->nodes_[node_id]->n_output());
      }
    }

    if (done_backward) {
      for_each(detail::g->operator_inputs_[node_id].begin(),
               detail::g->operator_inputs_[node_id].end(),
               [](auto &var) { run_backward(var); }); // sequential process
    }
  }
}
} // namespace trace

} // namespace kuu
