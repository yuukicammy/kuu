#include "functions.hpp"
#include "module.hpp"
#include "modules.hpp"
#include "optimizer.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <string>
#include <xtensor/xbuilder.hpp>

template <class OptimizerClass, typename OpmizerOptions>
void test_update(OpmizerOptions &&options) {
  struct net : public kuu::module {
    net()
        : conv1{kuu::conv_options<2>{3, 2, 3, true}}, conv2{2, 1, 3},
          l1{kuu::linear_options{9, 2, true}},
          l2{kuu::linear_options{2, 1, true}}, b1{2}, b2{1}, b3{2} {
      register_module("conv1", conv1);
      register_module("conv2", conv2);
      register_module("l1", l1);
      register_module("l2", l2);
      register_module("b1", b1);
      register_module("b2", b2);
      register_module("b3", b3);
    }
    kuu::tensor forward(const kuu::tensor &input, const kuu::tensor &gt) {
      // std::cout << kuu::shape2string(input.shape()) << std::endl;
      auto out = conv1->forward(input);
      // std::cout << kuu::shape2string(out.shape()) << std::endl;
      out = conv2->forward(kuu::function::relu::forward(b1->forward(out)));
      // std::cout << kuu::shape2string(out.shape()) << std::endl;
      out = l1->forward(kuu::function::relu::forward(b2->forward(out)));
      // std::cout << kuu::shape2string(out.shape()) << std::endl;
      out = l2->forward(kuu::function::relu::forward(b3->forward(out)));
      // std::cout << kuu::shape2string(out.shape()) << std::endl;
      return kuu::function::mean_squared_error::forward(out, gt);
    }
    kuu::conv2d conv1, conv2;
    kuu::linear l1, l2;
    kuu::batchnorm b1, b2, b3;
  };
  net n;
  std::unique_ptr<kuu::optimizer> optim =
      std::make_unique<OptimizerClass>(n.parameters(true), std::move(options));

  n.initialize(kuu::initializer::normal, 0, 1);

  kuu::tensor_type data = xt::ones<kuu::value_type>({3, 3, 7, 7});

  kuu::tensor input{data, false};
  input.set_grad(xt::zeros_like(data));
  kuu::tensor gt{xt::ones<float>({1, 1}) * 0.5, false};

  kuu::tensor loss = n.forward(input, gt);
  std::cout << optim->steps() << " , " << loss.data() << std::endl;
  while (0.0002 < loss.data()()) {
    loss.backward();
    std::cout << optim->steps() << " , " << loss.data() << std::endl;
    optim->update();
    optim->clear_grad();
    loss = n.forward(input, gt);
  }
  std::cout << optim->steps() << " , " << loss.data() << std::endl;
  return;
}

TEST(OptimizerTest, TestClearGrad) {
  kuu::tensor param0{xt::ones<float>({3, 3})};
  kuu::tensor param1{xt::ones<float>({3, 3}) * 2.};
  param0.set_grad(xt::ones<float>({3, 3}));
  param1.set_grad(xt::ones<float>({3, 3}));
  std::unique_ptr<kuu::optimizer> opt =
      std::make_unique<kuu::sgd>(std::vector<kuu::tensor>{param0, param1},
                                 kuu::sgd::options{0.1, 0.1, 0.1, 0.1, false});

  opt->clear_grad();
  ASSERT_EQ(param0.grad(), xt::zeros<float>({3, 3}));
  ASSERT_EQ(param1.grad(), xt::zeros<float>({3, 3}));
  ASSERT_EQ(param0.data(), xt::ones<float>({3, 3}));
  ASSERT_EQ(param1.data(), xt::ones<float>({3, 3}) * 2.);
}

TEST(OptimizerTest, TestSGDUpdate) {
  // test_update<kuu::sgd>(kuu::sgd::options{0.001, 0, 0, 0, false}); // 639
  // test_update<kuu::sgd>(kuu::sgd::options{0.001, 0.001, 0, 0, false}); //
  // 639 test_update<kuu::sgd>(kuu::sgd::options{0.001, 0.001, 0.9, 0,
  // false}); // 639
  // test_update<kuu::sgd>(kuu::sgd::options{0.001, 0.01, 0.9, 0.1, false});
  test_update<kuu::sgd>(kuu::sgd::options{0.001, 0.01, 0.9, 0, true});
}
