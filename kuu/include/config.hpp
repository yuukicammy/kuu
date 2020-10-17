#ifndef KUU_CONFIG_HPP
#define KUU_CONFIG_HPP

#include <string>
#include <xtensor/xarray.hpp>

namespace kuu {

using value_type = float;

template <typename T> class tensor_container;
using tensor_type = xt::xarray<value_type>;
using tensor = tensor_container<tensor_type>;

using id_type = std::string;

static constexpr bool kDebug = true;

const std::string klayout = "NCHW";

constexpr int kSeed = 0;

} // namespace kuu

#endif // KUU_CONFIG_HPP