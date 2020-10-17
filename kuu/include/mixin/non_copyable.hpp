#ifndef KUU_MIXIN_NON_COPYABLE_HPP
#define KUU_MIXIN_NON_COPYABLE_HPP

namespace kuu {
template <class Derived> class non_copyable {
protected:
  non_copyable() = default;
  ~non_copyable() = default;

public:
  non_copyable(const non_copyable &) = delete;
  Derived &operator=(const Derived &) = delete;
};
} // namespace kuu
#endif // KUU_MIXIN_NON_COPYABLE_HPP