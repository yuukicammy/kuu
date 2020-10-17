#ifndef KUU_FUNCTION_HPP
#define KUU_FUNCTION_HPP

#include "graph.hpp"
#include "tensor.hpp"
#include "util/util.hpp"
#include <cstddef>
#include <memory>
#include <string>

namespace kuu {

// for backward
class traceable_function {
public:
  traceable_function() = default;
  explicit traceable_function(const std::size_t n_output)
      : n_output_{n_output} {}
  ~traceable_function() = default;

  id_type id() const noexcept { return id_; }
  std::string name() const noexcept { return name_; }

  std::size_t n_output() const noexcept { return n_output_; }

  void set_name(const std::string name) noexcept { name_ = name; }

  std::function<void(const std::vector<tensor> &, std::vector<tensor> &)>
      backward_function;

protected:
  std::size_t n_output_;

private:
  id_type id_{"traceable_function-" + util::generate_id<id_type>()};
  std::string name_;
};

} // namespace kuu

#endif // KUU_FUNCTION_HPP
