#ifndef KUU_TENSOR_HPP
#define KUU_TENSOR_HPP

#include "config.hpp"
#include "util/converter.hpp"
#include "util/util.hpp"
#include <cstddef>
#include <memory>
#include <variant>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xscalar.hpp>
#include <xtensor/xstrided_view.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xsequence.hpp>

namespace kuu {
class graph;
class module;
namespace detail {
template <typename T> struct tensor_info;
extern std::shared_ptr<graph> g;
} // namespace detail

template <typename T> class tensor_container {

public:
  using tensor_type = T;
  using self_type = tensor_container<tensor_type>;
  using value_type = typename T::value_type;

  tensor_container();

  tensor_container(const tensor_container &) = default; // share, shallow copy
  tensor_container &
  operator=(const tensor_container &) = default; // share, shallow copy

  tensor_container(
      std::shared_ptr<detail::tensor_info<T>> ptr) // share, shallow copy
      : internal_{ptr} {}

  tensor_container(std::vector<std::size_t> &&shape,
                   const bool requires_grad = true);

  template <class XtensorType,
            typename = std::enable_if_t<std::is_base_of_v<
                xt::xexpression<std::remove_reference_t<XtensorType>>,
                std::remove_reference_t<XtensorType>>>>
  tensor_container(XtensorType &&data, const bool requires_grad = true);

  tensor_container(value_type scalar, const bool requires_grad = true);

  void backward();

  self_type clone() const;

  self_type share() {
    self_type shared{this->internal_};
    return shared;
  }

  void clear_grad();

  bool is_empty() const noexcept {
    return (this->shape().size() > 0) ? false : true;
  }

  // getter
  tensor_type cdata() const noexcept;
  tensor_type cgrad() const noexcept;
  std::vector<size_t> shape() const;
  bool requires_grad() const noexcept;
  std::size_t size() const noexcept;
  std::string name() const noexcept { return this->internal_->name; }
  id_type id() const noexcept { return this->internal_->id; }

  auto get() const noexcept { return this->internal_; }

  id_type creator_id() const noexcept { return this->internal_->creator_id; }

  tensor_type &data() { return this->internal_->data; }
  tensor_type &grad() { return this->internal_->grad; }

  std::size_t dim() const { return this->shape().size(); }

  // setter
  void set_creator_id(const id_type creator_id) {
    assert(this->internal_->creator_id == "");
    this->internal_->creator_id = creator_id;
  }
  void set_required_grad(const bool requires_grad) {
    assert(this->internal_);
    internal_->requires_grad = requires_grad;
  }
  void set_name(std::string name) {
    assert(this->internal_);
    this->internal_->name = name;
  }
  template <class XtensorType,
            typename = std::enable_if_t<std::is_base_of_v<
                xt::xexpression<std::remove_reference_t<XtensorType>>,
                std::remove_reference_t<XtensorType>>>>
  void set_grad(XtensorType &&grad);

  // operator
  template <class XtensorType,
            typename = std::enable_if_t<std::is_base_of_v<
                xt::xexpression<std::remove_reference_t<XtensorType>>,
                std::remove_reference_t<XtensorType>>>>
  tensor_container &operator=(XtensorType &&);
  tensor_container operator[](const size_t i) const;
  friend std::ostream &operator<<(std::ostream &os, const self_type &obj) {
    os << obj.cdata();
    return os;
  }

  std::vector<value_type> as_vector() const {
    std::vector<value_type> v;
    for_each(this->data().storage().cbegin(), this->data().storage().cend(),
             [&v](const auto &val) { v.push_back(val); });
    return v;
  }

private:
  std::shared_ptr<detail::tensor_info<T>> internal_;
};

namespace detail {
template <typename T> struct tensor_info {
  tensor_info() : id{"tensor-" + util::generate_id<id_type>()} {};
  T data;
  T grad;
  std::vector<std::size_t> shape;
  id_type creator_id; // function id
  bool requires_grad;
  std::string name;
  std::string id;
};
} // namespace detail

template <typename T>
inline tensor_container<T>::tensor_container()
    : internal_{std::make_shared<detail::tensor_info<T>>()} {}

template <typename T>
tensor_container<T>::tensor_container(std::vector<std::size_t> &&shape,
                                      const bool requires_grad)
    : internal_{std::make_shared<detail::tensor_info<T>>()} {
  this->internal_->data = T(shape);
  this->internal_->grad = xt::zeros<typename T::value_type>(shape);
  this->internal_->shape = std::move(shape);
  this->internal_->requires_grad = requires_grad;
}

template <typename T>
template <class XtensorType, typename>
tensor_container<T>::tensor_container(XtensorType &&data,
                                      const bool requires_grad)
    : internal_{std::make_shared<detail::tensor_info<T>>()} {
  this->internal_->shape =
      std::vector<size_t>(data.shape().begin(), data.shape().end());
  this->internal_->data = std::forward<XtensorType>(data);
  this->internal_->grad = xt::zeros_like(this->internal_->data);
  this->internal_->requires_grad = requires_grad;
}

template <typename T>
tensor_container<T>::tensor_container(value_type scalar,
                                      const bool requires_grad)
    : internal_{std::make_shared<detail::tensor_info<T>>()} {
  this->internal_->shape = std::vector<size_t>{1};
  this->internal_->data = xt::xscalar{scalar};
  this->internal_->grad = xt::zeros_like(this->internal_->data);
  this->internal_->requires_grad = requires_grad;
}

template <typename T> T tensor_container<T>::cdata() const noexcept {
  assert(this->internal_);
  return this->internal_->data;
}

template <typename T> T tensor_container<T>::cgrad() const noexcept {
  assert(this->internal_);
  return this->internal_->grad;
}

template <typename T> std::vector<size_t> tensor_container<T>::shape() const {
  assert(this->internal_);
  assert(this->internal_->data.shape() == this->internal_->grad.shape());
  assert(this->internal_->data.shape().size() == this->internal_->shape.size());
  auto s = this->internal_->data.shape();
  for (std::size_t i = 0; i < this->internal_->shape.size(); i++) {
    assert(this->internal_->shape[i] == s[i]);
  }
  return this->internal_->shape;
}

template <typename T> bool tensor_container<T>::requires_grad() const noexcept {
  assert(this->internal_);
  return this->internal_->requires_grad;
}

template <typename T> std::size_t tensor_container<T>::size() const noexcept {
  assert(this->internal_);
  if (this->is_empty()) {
    return 0;
  }
  size_t n = 1;
  for (auto s : this->shape()) {
    n *= s;
  }
  return n;
}

template <typename T> void tensor_container<T>::clear_grad() {
  this->internal_->grad = xt::zeros_like(this->internal_->data);
}

template <typename T> tensor_container<T> tensor_container<T>::clone() const {
  tensor_container<T> copy{this->cdata(), this->requires_grad()};
  copy.set_grad(this->cgrad());
  copy.set_creator_id(this->creator_id());
  copy.set_name(this->name());
  return copy;
};

template <typename T>
template <class XtensorType, typename>
void tensor_container<T>::set_grad(XtensorType &&grad) {
  this->internal_->grad = std::forward<XtensorType>(grad);
  this->shape();
}

template <typename T>
template <class XtensorType, typename>
inline tensor_container<T> &tensor_container<T>::operator=(XtensorType &&e) {
  this->internal_->data = std::forward<XtensorType>(e);
  this->shape();
  return *this;
}

template <typename T>
inline tensor_container<T>
    tensor_container<T>::operator[](const size_t i) const {
  T stride = xt::strided_view(this->internal_->data, {i, xt::ellipsis()});
  tensor_container<T> clone{std::move(stride), this->internal_->requires_grad};
  clone.set_grad(xt::strided_view(this->internal_->grad, {i, xt::ellipsis()}));
  return clone;
}

namespace trace {
void run_backward(const tensor &root);
}

template <typename T> void tensor_container<T>::backward() {
  // std::cout << "tensor::backward" << std::endl;
  set_grad(xt::ones_like(internal_->data));
  trace::run_backward(*this);
}

} // namespace kuu
#endif // KUU_TENSOR_HPP