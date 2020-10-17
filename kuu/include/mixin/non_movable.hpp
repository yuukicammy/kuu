#ifndef KUU_MIXIN_NON_MOVABLE_HPP
#define KUU_MIXIN_NON_MOVABLE_HPP

namespace kuu {
template <class Derived> class non_movable {
public:
  non_movable(non_movable &&) = delete;
  Derived &operator=(Derived &&) = delete;

protected:
  non_movable() = default;
  ~non_movable() = default;
};
} // namespace kuu
#endif // KUU_MIXIN_NON_MOVABLE_HPP