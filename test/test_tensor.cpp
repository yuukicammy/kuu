#include "tensor.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xrandom.hpp>

TEST(TensorTest, TensorShape) {
  kuu::tensor t{{5, 10}}; // set shape
  std::vector<std::size_t> shape = {5, 10};
  ASSERT_EQ(t.shape(), shape);
}

TEST(TensorTest, TensorSize) {
  kuu::tensor t{{5, 10}}; // set shape
  ASSERT_EQ(t.size(), 50);
}

TEST(TensorTest, TensorName) {
  kuu::tensor t{{1, 1}};
  t.set_name("name");
  ASSERT_EQ(t.name(), "name");
}

TEST(TensorTest, TensorCreatorId) {
  kuu::tensor t{{1, 1}};
  auto creator_id = kuu::util::generate_id<>();
  t.set_creator_id(creator_id);
  ASSERT_EQ(t.creator_id(), creator_id);
}

TEST(TensorTest, TensorGrad) {
  kuu::tensor t{{3, 3}};
  xt::xarray<kuu::value_type> grad;
  if (std::is_same_v<kuu::value_type, int>) {
    grad = xt::random::randint<int>(t.shape(), -10, 10);
  } else {
    grad = xt::random::rand<kuu::value_type>(t.shape(), -10, 10);
  }
  t.set_grad(grad);
  ASSERT_EQ(t.grad(), grad);
  ASSERT_EQ(t.cgrad(), grad);
}

TEST(TensorTest, TensorData) {
  xt::xarray<kuu::value_type> data;
  if (std::is_same_v<kuu::value_type, int>) {
    data = xt::random::randint<int>({3, 3}, -10, 10);
  } else {
    data = xt::random::rand<kuu::value_type>({3, 3}, -10, 10);
  }
  kuu::tensor t{data, false};
  ASSERT_EQ(t.data(), data);
  ASSERT_EQ(t.cdata(), data);
  ASSERT_EQ(t.requires_grad(), false);

  kuu::tensor u{{3, 3}};
  u = data;
  ASSERT_EQ(u.data(), data);
  ASSERT_EQ(u.cdata(), data);
}

TEST(TensorTest, TensorCopy) {
  kuu::tensor src{xt::ones<kuu::value_type>({3, 3}), true};

  src.set_grad(xt::ones_like(src.data()));
  src.set_creator_id(kuu::util::generate_id<>());
  src.set_name("tensor!");

  kuu::tensor dst1 = src; // shallow copy
  kuu::tensor dst2{src};  // shallow copy

  TENSOR_EQ(dst1, src);
  TENSOR_EQ(dst2, src);

  src.set_grad(xt::zeros_like(src.data()));
  src.set_name("tensor?");

  TENSOR_EQ(dst1, src);
  TENSOR_EQ(dst2, src);
}

TEST(TensorTest, TensorStride) {
  xt::xarray<kuu::value_type> data = {{1, 2}, {3, 4}};
  xt::xarray<kuu::value_type> grad = {{5, 6}, {7, 8}};
  kuu::tensor t{data};
  t.set_grad(grad);

  kuu::tensor stride = t[0];
  TENSOR_CLONE_EQ(stride, t[0]);

  xt::xarray<kuu::value_type> val = {1, 2};
  ASSERT_EQ(stride.cdata(), val);
  ASSERT_EQ(stride.data(), val);

  xt::xarray<kuu::value_type> val1 = {5, 6};
  ASSERT_EQ(stride.cgrad(), val1);
  ASSERT_EQ(stride.grad(), val1);
}