#ifndef KUU_TEST_COMMON_HPP
#define KUU_TEST_COMMON_HPP

#include "gtest/gtest.h"

#define TENSOR_EQ(tensor0, tensor1)                                            \
  ASSERT_EQ(tensor0.data().storage(), tensor1.data().storage());               \
  ASSERT_EQ(tensor0.grad().storage(), tensor1.grad().storage());               \
  ASSERT_EQ(tensor0.grad().shape(), tensor1.grad().shape());                   \
  ASSERT_EQ(tensor0.data().shape(), tensor1.data().shape());                   \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  ASSERT_EQ(tensor0.name(), tensor1.name());                                   \
  ASSERT_EQ(tensor0.requires_grad(), tensor1.requires_grad());                 \
  ASSERT_EQ(tensor0.id(), tensor1.id());

#define TENSOR_CLONE_EQ(tensor0, tensor1)                                      \
  ASSERT_EQ(tensor0.data().storage(), tensor1.data().storage());               \
  ASSERT_EQ(tensor0.grad().storage(), tensor1.grad().storage());               \
  ASSERT_EQ(tensor0.grad().shape(), tensor1.grad().shape());                   \
  ASSERT_EQ(tensor0.data().shape(), tensor1.data().shape());                   \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  ASSERT_EQ(tensor0.name(), tensor1.name());                                   \
  ASSERT_EQ(tensor0.requires_grad(), tensor1.requires_grad());                 \
  ASSERT_TRUE(tensor0.id() != tensor1.id());

template <typename T0, typename T1> void CLOSE_ALL(T0 &&x0, T1 &&x1) {
  auto itr0 = x0.cbegin();
  auto itr1 = x1.cbegin();
  for (; itr0 != x0.cend() && itr1 != x1.cend(); itr0++, itr1++) {
    EXPECT_FLOAT_EQ(*itr0, *itr1);
  }
}

template <typename T0, typename T1>
void CLOSE_ALL(T0 &&x0, T1 &&x1, double abs_error) {
  auto itr0 = x0.cbegin();
  auto itr1 = x1.cbegin();
  for (; itr0 != x0.cend() && itr1 != x1.cend(); itr0++, itr1++) {
    EXPECT_NEAR(*itr0, *itr1, abs_error);
  }
}

#define XTENSOR_EQ(tensor0, tensor1)                                           \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  ASSERT_EQ(tensor0.storage(), tensor1.storage());

#define XTENSOR_CLOSE(tensor0, tensor1)                                        \
  ASSERT_EQ(tensor0.shape(), tensor1.shape());                                 \
  CLOSE_ALL(tensor0, tensor1);

#endif // KUU_TEST_COMMON_HPP
