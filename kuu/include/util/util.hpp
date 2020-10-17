#ifndef KUU_UTIL_UTIL_HPP
#define KUU_UTIL_UTIL_HPP

#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace kuu {

namespace util {
template <typename Head> bool requires_grad(Head &&tensor) {
  return tensor.requires_grad();
}

template <typename Head, typename... Tail>
bool requires_grad(Head &&head, Tail &&... tail) {
  return head.requires_grad() +
         util::requires_grad(std::forward<Tail>(tail)...);
}

template <typename Map, typename Key>
bool find(const Map &map, const Key &key) {
  return map.find(key) != map.end() ? true : false;
}

template <typename IdType = std::string> inline IdType generate_id() {
  return boost::lexical_cast<IdType>(boost::uuids::random_generator()());
}

} // namespace util
} // namespace kuu

#endif // KUU_UTIL_UTIL_HPP