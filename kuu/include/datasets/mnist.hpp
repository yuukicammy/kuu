#ifndef KUU_DATASETS_MNIST_HPP
#define KUU_DATASETS_MNIST_HPP

#include "config.hpp"
#include "tensor.hpp"
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <istream>
#include <random>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xtensor.hpp>

namespace kuu {
enum mode_type { train = 0, test = 1 };

namespace data {
class dataset {
public:
private:
};

class MNIST {
public:
  MNIST() = default;
  MNIST(std::string train_image_file, std::string train_label_file,
        std::string test_image_file, std::string test_label_file,
        unsigned int seed = 0)
      : train_image_file_{train_image_file},
        train_label_file_{train_label_file}, test_image_file_{test_image_file},
        test_label_file_{test_label_file}, rand_generator_{seed},
        distribution_train_{0, n_train_}, distribution_test_{0, n_test_} {
    load_all();
  }

  xt::xtensor<value_type, 4> load_images(std::string file_name,
                                         std::size_t n_images) {
    std::cout << "Loading " << file_name << "..." << std::endl;
    std::ifstream ifs(file_name.c_str(), std::ios::in | std::ios::binary);
    assert(ifs.is_open());
    assert(ifs);

    char a[4];

    // magic number
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;

    // number of images
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;
    // assert(n_images == 60000 || n_images == 10000);

    // number of rows
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;
    assert(char4toint32(a) == height_);

    // number of cols
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;
    assert(char4toint32(a) == width_);

    std::array<std::size_t, 4> shape = {n_images, 1, height_, width_};
    xt::xtensor<value_type, 4> data{shape};

    std::array<std::size_t, 2> s = {height_, width_};
    std::array<unsigned char, width_ * height_> image;
    for (std::size_t i = 0; i < n_images; i++) {
      ifs.read(reinterpret_cast<char *>(&image), width_ * height_);
      xt::view(data, i, 0, xt::all(), xt::all()) =
          xt::adapt(reinterpret_cast<unsigned char *>(&image), sizeof(image),
                    xt::no_ownership(), s);
      // std::cout << xt::view(data, i, 0, xt::all(), xt::all()) << std::endl;
    }
    data /= 255.;
    return data;
  }

  template <std::size_t N>
  xt::xtensor<value_type, 1> load_labels(std::string file_name) {
    std::cout << "Loading " << file_name << "..." << std::endl;
    std::ifstream ifs(file_name.c_str(), std::ios::in | std::ios::binary);
    assert(ifs.is_open());
    assert(ifs);

    char a[4];

    // magic number
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;

    // number of data
    ifs.read(a, 4);
    // std::cout << char4toint32(a) << std::endl;

    char labels[N];
    ifs.read((char *)labels, N);
    std::array<std::size_t, 1> shape{N};
    xt::xtensor<value_type, 1> data =
        xt::adapt((char *)labels, N, xt::no_ownership(), shape);

    return data;
  }

  void load_all() {
    train_images_ = load_images(train_image_file_, n_train_);
    train_labels_ = load_labels<n_train_>(train_label_file_);
    test_images_ = load_images(test_image_file_, n_test_);
    test_labels_ = load_labels<n_test_>(test_label_file_);
  }

  std::pair<xt::xtensor<value_type, 4>, xt::xtensor<value_type, 1>>
  load(mode_type mode, std::size_t N) {
    std::array<std::size_t, 4> ishape{N, 1, height_, width_};
    xt::xtensor<value_type, 4> image_batch{ishape};

    std::array<std::size_t, 1> lshape{N};
    xt::xtensor<value_type, 1> label_batch{lshape};

    auto &images = mode == mode_type::train ? train_images_ : test_images_;
    auto &labels = mode == mode_type::train ? train_labels_ : test_labels_;

    for (std::size_t i = 0; i < N; i++) {
      int id = mode == mode_type::train ? distribution_train_(rand_generator_)
                                        : distribution_test_(rand_generator_);
      xt::view(image_batch, i, xt::all(), xt::all(), xt::all()) =
          xt::eval(xt::view(images, id, xt::all(), xt::all(), xt::all()));
      label_batch(i) = xt::view(labels, id);
    }

    return std::make_pair(image_batch, label_batch);
  }

  std::pair<xt::xtensor<value_type, 4>, xt::xtensor<value_type, 1>>
  load(mode_type mode, std::size_t N, std::size_t batch_id) {
    size_t batch_size = N;
    if (mode == mode_type::test && N * (batch_id + 1) >= n_test_) {
      batch_size = n_test_ % N;
    }

    std::array<std::size_t, 4> ishape{batch_size, 1, height_, width_};
    xt::xtensor<value_type, 4> image_batch{ishape};

    std::array<std::size_t, 1> lshape{batch_size};
    xt::xtensor<value_type, 1> label_batch{lshape};

    auto &images = mode == mode_type::train ? train_images_ : test_images_;
    auto &labels = mode == mode_type::train ? train_labels_ : test_labels_;

    for (std::size_t i = 0; i < batch_size; i++) {
      int id;
      if (mode == mode_type::train) {
        id = distribution_train_(rand_generator_);
      } else {
        id = N * batch_id + i;
      }
      xt::view(image_batch, i, xt::all(), xt::all(), xt::all()) =
          xt::eval(xt::view(images, id, xt::all(), xt::all(), xt::all()));
      label_batch(i) = xt::view(labels, id);
    }

    return std::make_pair(image_batch, label_batch);
  }

  void normalize(double mean, double std) {
    std::cout << xt::mean(train_images_) << std::endl;
    train_images_ = (train_images_ - mean) / std;
    std::cout << xt::mean(train_images_) << std::endl;
    test_images_ = (test_images_ - mean) / std;
  }

  size_t size(mode_type mode) {
    if (mode == mode_type::train) {
      return n_train_;
    }
    return n_test_;
  }

private:
  int32_t char4toint32(const char *c) {
    int32_t x = (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];
    return x;
  }

  std::default_random_engine rand_generator_;
  std::uniform_int_distribution<int> distribution_test_;
  std::uniform_int_distribution<int> distribution_train_;

  std::string train_image_file_;
  std::string train_label_file_;
  std::string test_image_file_;
  std::string test_label_file_;

  static constexpr std::size_t height_ = 28;
  static constexpr std::size_t width_ = 28;
  static constexpr std::size_t n_train_ = 60000;
  static constexpr std::size_t n_test_ = 10000;

  xt::xtensor<value_type, 4> train_images_;
  xt::xtensor<value_type, 1> train_labels_;

  xt::xtensor<value_type, 4> test_images_;
  xt::xtensor<value_type, 1> test_labels_;
};
} // namespace data
} // namespace kuu
#endif // KUU_DATASETS_MNIST_HPP
