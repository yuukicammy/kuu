#ifndef KUU_EXARRAY_HPP
#define KUU_EXARRAY_HPP

#include "config.hpp"
#include "tensor.hpp"
#include <xtensor/xadapt.hpp>

namespace kuu {
template <std::size_t N, typename T = std::size_t> class exarray {
public:
  exarray(T &&data) { data_.fill(data); }
  exarray(std::array<T, N> data) : data_{std::move(data)} {}
  exarray(std::initializer_list<T> list) {
    if (list.size() == 1 && 1 < N) {
      data_.fill(*list.begin());
    }
    std::copy(list.begin(), list.end(), data_.begin());
  }
  exarray(const tensor &t) {
    if (t.size() == 1) {
      data_.fill(*t.cdata().begin());
    } else {
      assert(t.size() == N);
      for (std::size_t i = 0; i < N; i++) {
        data_[i] = static_cast<T>(t.cdata().at(i));
      }
    }
  }

  template <std::size_t I> T get() {
    static_assert(I <= N);
    return data_[I];
  }

  tensor asTensor() const {
    xt::xarray<tensor::value_type> data = xt::adapt(data_);
    tensor t{std::move(data), false};
    return t;
  }

  template <typename T1> inline exarray &operator=(T1 &&val) {
    data_.fill(val);
    return *this;
  }

private:
  std::array<T, N> data_;
};
} // namespace kuu

#endif // KUU_EXARRAY_HPP