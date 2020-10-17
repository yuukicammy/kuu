#ifndef KUU_MODULE_HPP
#define KUU_MODULE_HPP

#include "graph.hpp"
#include "initializer.hpp"
#include "tensor.hpp"
#include <execution>
#include <map>
#include <memory>
#include <unordered_map>

namespace kuu {

class module;

class any_module {
  template <typename ModuleType>
  explicit any_module(std::shared_ptr<ModuleType> module) {}
};

template <class ModuleType> class module_holder {
public:
  using module_type = ModuleType;
  using shared_module = std::shared_ptr<module_type>;
  using self_type = module_holder<module_type>;

  explicit module_holder() : pmodule_(new ModuleType()) {}

  module_holder(std::shared_ptr<ModuleType> p) : pmodule_(p) {}

  template <typename Head, typename... Tail>
  explicit module_holder(Head &&head, Tail &&... tail)
      : pmodule_(new ModuleType(std::forward<Head>(head),
                                std::forward<Tail>(tail)...)) {}

  template <typename Head>
  explicit module_holder(Head &&head)
      : pmodule_(new ModuleType(std::forward<Head>(head))) {}

  shared_module operator->() { return this->pmodule_; }

  ModuleType operator*() const { return *(this->pmodule_); }

  shared_module ptr() { return pmodule_; }

  virtual void pretty_print(std::ostream &stream) const {}

private:
  std::shared_ptr<ModuleType> pmodule_;
};

class module : public std::enable_shared_from_this<module> {
  friend class graph;

public:
  module()
      : id_{"module-" + util::generate_id<id_type>()}, is_training_{true},
        is_initialized_{false} {}
  ~module() = default;

  inline std::string id() const noexcept { return id_; }
  inline bool is_training() const noexcept { return is_training_; }
  inline bool is_initialized() const noexcept { return is_initialized_; }

  template <class F, typename... U>
  void initialize(F initializer_function, U... args);
  template <class F> void initialize(F initializer_function);

  void clear_grad(const bool recursive = true);
  void set_name(std::string name) { name_ = name; }
  void train(bool on = true);

  std::vector<tensor> parameters(bool recursive = true);
  std::optional<tensor> parameter(std::string name) {
    if (util::find(params_, name)) {
      return this->params_[name];
    } else {
      return std::nullopt;
    }
  }

protected:
  template <class ModuleType>
  std::shared_ptr<ModuleType> register_module(std::string name,
                                              module_holder<ModuleType> holder);

  tensor &register_parameter(std::string name, tensor param,
                             bool requires_grad = true);

  std::unordered_map<std::string, tensor> params_;
  std::map<std::string, std::shared_ptr<module>> submodules_;
  bool is_initialized_;

private:
  id_type id_;
  std::string name_;
  bool is_training_;
}; // namespace kuu

template <class F, class... U>
void module::initialize(F initializer_function, U... args) {
  // std::cout << "initialization with args in module. " << id() << std::endl;
  if (!this->is_initialized()) {
    std::for_each(std::execution::par_unseq, std::begin(params_),
                  std::end(params_),
                  [initializer_function, args...](auto &param) {
                    initializer_function(param.second, args...);
                    if (kDebug && 0) {
                      std::cout << "initialized parameter." << std::endl;
                      std::cout << param.second.data() << std::endl;
                    }
                  });
  }

  std::for_each(std::execution::par_unseq, std::begin(submodules_),
                std::end(submodules_),
                [initializer_function, args...](auto &submodule) {
                  if (!submodule.second->is_initialized()) {
                    submodule.second->initialize(initializer_function, args...);
                  }
                });
}

template <class F> void module::initialize(F initializer_function) {
  // std::cout << "initialization in module. " << id() << std::endl;
  std::for_each(std::execution::par_unseq, std::begin(params_),
                std::end(params_), [initializer_function](auto &param) {
                  initializer_function(param.second);
                  if (kDebug && 0) {
                    std::cout << "initialized parameter." << std::endl;
                    std::cout << param.second.data() << std::endl;
                  }
                });
  std::for_each(std::execution::par_unseq, std::begin(submodules_),
                std::end(submodules_), [initializer_function](auto &submodule) {
                  submodule.second->initialize(initializer_function);
                });
}

template <class ModuleType>
std::shared_ptr<ModuleType>
module::register_module(std::string name, module_holder<ModuleType> holder) {
  // std::cout << "register module : " << name << std::endl;
  assert(holder.ptr());
  holder->set_name(name);
  this->submodules_[name] = holder.ptr();

  return holder.ptr();
}

tensor &module::register_parameter(std::string name, tensor param,
                                   bool requires_grad) {
  param.set_required_grad(requires_grad);
  param.set_name(name);
  this->params_[name] = param;

  assert(param.id() == this->params_[name].id());
  return this->params_[name];
}

std::vector<tensor> module::parameters(bool recursive) {
  std::vector<tensor> parameters;
  std::for_each(
      std::begin(params_), std::end(params_),
      [&parameters](auto &param) { parameters.emplace_back(param.second); });
  if (recursive) {
    std::for_each(std::begin(submodules_), std::end(submodules_),
                  [&parameters](auto &submodule) {
                    auto v = submodule.second->parameters(true);
                    std::copy(v.begin(), v.end(),
                              std::back_inserter(parameters));
                  });
  }
  return parameters;
}

void module::clear_grad(bool recursive) {
  std::for_each(std::execution::par_unseq, std::begin(params_),
                std::end(params_),
                [](std::pair<const std::string, tensor> &param) {
                  param.second.clear_grad();
                });
  if (recursive) {
    std::for_each(std::execution::par_unseq, std::begin(submodules_),
                  std::end(submodules_), [recursive](auto &submodule) {
                    submodule.second->clear_grad(recursive);
                  });
  }
}

void module::train(bool on) {
  std::for_each(std::begin(params_), std::end(params_),
                [on](std::pair<const std::string, tensor> &param) {
                  param.second.set_required_grad(on);
                });
  for (auto &sub : submodules_) {
    sub.second->train(on);
  }
  is_training_ = on;
}

} // namespace kuu

#endif // KUU_MODULE_HPP