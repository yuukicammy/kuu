#include "datasets/mnist.hpp"
#include "cxxopts.hpp"
#include "functions.hpp"
#include "initializer.hpp"
#include "module.hpp"
#include "modules.hpp"
#include "optimizer.hpp"
#include "optimizers/sgd.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xsort.hpp>

cxxopts::ParseResult parse(int argc, char *argv[]) {
  try {
    cxxopts::Options options(argv[0], "kuu test with mnist");
    options.positional_help("[optional args]").show_positional_help();

    options.allow_unrecognised_options().add_options()(
        "h,help", "Show brief usage message")(
        "d,dataset", "The dataset directory path to use ",
        cxxopts::value<std::string>()->default_value("data/boston-housing"))(
        "b,batchsize", "The number of data in each mini-batch",
        cxxopts::value<size_t>()->default_value("16"))(
        "l,learnrate", "The learning rate of optimization",
        cxxopts::value<float>()->default_value("0.0001"))(
        "e,epoch", "The number of sweeps over the dataset to train",
        cxxopts::value<size_t>()->default_value("100000000"))(
        "o,output", "The directory to output the result",
        cxxopts::value<std::string>()->default_value("./result"));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
    return result;
  } catch (const cxxopts::OptionException &e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

bool equal(const kuu::tensor &t) {
  xt::xarray<float> &&data = t.cdata();
  data.reshape({(int)data.shape()[0], -1});
  auto &&x0 = xt::view(data, 0, xt::all());
  auto &&x1 = xt::view(data, 1, xt::all());

  for (std::size_t i = 0; i < x0.size(); i++) {
    if (1e-6 < std ::fabs(x0(i) - x1(i))) {
      return false;
    }
  }
  return true;
}

struct net : public kuu::module {
  net()
      : conv1{kuu::conv_options<2>{1, 8, 5, 1, 2}}, conv2{kuu::conv_options<2>{
                                                        8, 1, 5, 1, 2}},
        linear1{kuu::linear_options{784, 10, false}}, bn1{8}, bn2{1} {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("linear1", linear1);
    register_module("bn1", bn1);
    register_module("bn2", bn2);
  }

  kuu::tensor forward(const kuu::tensor &input) {
    auto out = conv1->forward(input);
    out = bn1->forward(out);
    out = kuu::function::relu::forward(out);
    out = conv2->forward(out);
    out = bn2->forward(out);
    out = kuu::function::relu::forward(out);
    out = linear1->forward(out);
    return out;
  }

  kuu::tensor training(const kuu::tensor &input, const kuu::tensor &gt) {
    this->train(true);
    auto out = forward(input);
    return kuu::function::softmax_cross_entropy::forward(out, gt);
  }

  kuu::tensor predict(const kuu::tensor &input) {
    this->train(false);
    auto out = forward(input);
    auto pred = xt::argmax(out.data(), {1});
    assert(pred.dimension() == 1);
    return kuu::tensor{std::move(pred), false};
  }

  double accuracy(const kuu::tensor &pred, const kuu::tensor &gt) {
    auto &&p = pred.cdata();
    auto &&g = gt.cdata();
    auto acc = xt::mean(xt::equal(p, g));
    return acc();
  }

  kuu::conv2d conv1, conv2;
  kuu::linear linear1;
  kuu::batchnorm bn1, bn2;
};

float evaluate(net &n, kuu::data::MNIST &mnist, size_t batch_size) {
  double acc_sum = 0.0;
  int num = 0;
  for (size_t i = 0; i < (mnist.size(kuu::mode_type::test) / batch_size) - 1;
       i++) {
    auto test_batch = mnist.load(kuu::mode_type::test, batch_size, i);
    kuu::tensor test_data{test_batch.first, false};
    kuu::tensor test_gt{test_batch.second, false};
    auto pred = n.predict(test_data);
    acc_sum += n.accuracy(pred, test_gt) * test_data.shape()[0];
    num += test_data.shape()[0];
  }
  float acc = static_cast<float>(acc_sum / static_cast<double>(num));
  return acc;
}

int main(int argc, char *argv[]) {

  auto result = parse(argc, argv);

  const std::size_t epoch = result["epoch"].as<std::size_t>();
  const std::size_t batch_size = result["batchsize"].as<std::size_t>();
  const std::string data_dir = result["dataset"].as<std::string>();
  kuu::data::MNIST mnist{data_dir + "/train-images-idx3-ubyte",
                         data_dir + "/train-labels-idx1-ubyte",
                         data_dir + "/t10k-images-idx3-ubyte",
                         data_dir + "/t10k-labels-idx1-ubyte"};

  // mnist.normalize(0.1307, 0.3081);
  net n;
  std::unique_ptr<kuu::optimizer> optim = std::make_unique<kuu::sgd>(
      n.parameters(true),
      kuu::sgd::options{result["learnrate"].as<float>(), 0.0001, 0.9, false});

  n.initialize(kuu::initializer::he_normal);

  double sloss = 0;
  for (std::size_t i = 0; i < epoch; i++) {
    auto batch = mnist.load(kuu::mode_type::train, batch_size);
    kuu::tensor input{batch.first, false};
    kuu::tensor gt{batch.second, false};
    optim->clear_grad();
    kuu::tensor loss = n.training(input, gt);
    sloss += loss.data()();
    loss.backward();
    optim->update();
    if (i % 100 == 0) {
      if (i == 0) {
        std::cout << "loss: " << optim->steps() << " , " << sloss * 0.1
                  << std::endl;
      } else {
        std::cout << "loss: " << optim->steps() << " , " << sloss * 0.1
                  << std::endl;
        sloss = 0;
      }
      float acc = evaluate(n, mnist, batch_size);
      std::cout << "test acc: " << acc << std::endl;
    }
  }

  return 0;
}