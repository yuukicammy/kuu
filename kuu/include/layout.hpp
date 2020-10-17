#ifndef KUU_LAYOUT_HPP
#define KUU_LAYOUT_HPP

namespace kuu {
struct NHWC {
  static constexpr uint16_t N = 0;
  static constexpr uint16_t H = 1;
  static constexpr uint16_t W = 2;
  static constexpr uint16_t C = 3;
};

struct NCHW {
  static constexpr uint16_t N = 0;
  static constexpr uint16_t C = 1;
  static constexpr uint16_t H = 2;
  static constexpr uint16_t W = 3;
};

} // namespace kuu
#endif //  KUU_LAYOUT_HPP