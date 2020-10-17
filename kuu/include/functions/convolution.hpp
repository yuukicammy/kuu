#ifndef KUU_FUNCTIONS_CONVOLUTION_HPP
#define KUU_FUNCTIONS_CONVOLUTION_HPP

#include "convolution.hpp"
#include "exarray.hpp"
#include "function.hpp"
#include "layout.hpp"
#include <cassert>
#include <execution>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xmath.hpp>

namespace kuu {

template <typename T0, typename T1>
xt::xarray<tensor::value_type>
im2col(T0 &&x, T1 &&weight_shape, exarray<2> stride = 1, exarray<2> padding = 0,
       exarray<2> dilation = 1) {
  auto x_shape = x.shape();
  std::size_t W_f = weight_shape[3];
  std::size_t H_f = weight_shape[2];
  std::size_t C_in = weight_shape[1];
  std::size_t C_out = weight_shape[0];

  assert(x_shape[NCHW::C] == C_in);

  // zero-padding
  x_shape[NCHW::H] += 2 * padding.get<0>();
  x_shape[NCHW::W] += 2 * padding.get<1>();
  xt::xarray<tensor::value_type> padx = xt::zeros<tensor::value_type>(x_shape);
  xt::view(padx, xt::all(), xt::all(),
           xt::range(padding.get<0>(), x.shape()[NCHW::H] + padding.get<0>()),
           xt::range(padding.get<1>(), x.shape()[NCHW::W] + padding.get<1>())) =
      x;

  assert(x_shape[NCHW::H] % stride.get<0>() == 0);
  assert(x_shape[NCHW::W] % stride.get<1>() == 0);

  std::size_t N = x_shape[NCHW::N];
  std::size_t H_out = (x_shape[NCHW::H] - H_f) / stride.get<0>() + 1;
  std::size_t W_out = (x_shape[NCHW::W] - W_f) / stride.get<1>() + 1;
  std::size_t Cols = x_shape[NCHW::C] * H_f * W_f;

  std::vector<std::size_t> im2col_shape = {N, H_out, W_out, Cols};

  xt::xarray<value_type> im2col = xt::zeros<value_type>(im2col_shape);

  for (int i = 0; i <= x_shape[NCHW::H] - H_f; i += stride.get<0>()) {
    for (int j = 0; j <= x_shape[NCHW::W] - W_f; j += stride.get<1>()) {
      // extract all data in filter window
      auto v = xt::eval(xt::view(padx, xt::all(), xt::all(),
                                 xt::range(i, i + H_f), xt::range(j, j + W_f)));

      // flatten
      v.reshape({N, Cols});

      xt::view(im2col, xt::all(), i / stride.get<0>(), j / stride.get<1>(),
               xt::all()) = v;
    }
  }
  im2col.reshape({N * H_out * W_out, Cols});
  return im2col;
}

template <typename T0, typename T1, typename T2>
xt::xarray<tensor::value_type>
col2im(T0 &&col, T1 &&x_shape, T2 &&weight_shape, exarray<2> stride = 1,
       exarray<2> padding = 0, exarray<2> dilation = 1) {

  int Cols = col.shape()[1];
  int C_out = weight_shape[0];
  int C_in = weight_shape[1];
  int H_f = weight_shape[2];
  int W_f = weight_shape[3];

  // std::cout << "x_shape: " << shape2string(x_shape) << std::endl;

  std::vector<std::size_t> padx_shape{x_shape.begin(), x_shape.end()};
  padx_shape[NCHW::H] += 2 * padding.get<0>();
  padx_shape[NCHW::W] += 2 * padding.get<1>();

  int H_out = (padx_shape[NCHW::H] - H_f) / stride.get<0>() + 1;
  int W_out = (padx_shape[NCHW::W] - W_f) / stride.get<1>() + 1;

  xt::xarray<value_type> padx = xt::zeros<value_type>(padx_shape);

  // std::cout << "padx_shape: " << shape2string(padx_shape) << std::endl;

  col.reshape(
      {-1, H_out, W_out, C_in, H_f, W_f}); // {N, H_out, W_out, C_in, H_f, W_f}

  // std::cout << "col_shape: " << shape2string(col.shape()) << std::endl;
  assert(col.shape()[0] == x_shape[0]);
  col = xt::transpose(col,
                      {0, 3, 4, 5, 1, 2}); // {N, C_in, H_f, W_f, H_out, W_out}
  // std::cout << "col_shape: " << shape2string(col.shape()) << std::endl;

  int N = col.shape()[0];
  for (int i = 0; i < H_f; i++) {
    int i_max = i + stride.get<0>() * H_out;
    for (int j = 0; j < W_f; j++) {
      int j_max = j + stride.get<1>() * W_out;
      /*
      auto clopped_padx = xt::view(padx, xt::all(), xt::all(),
                                   xt::range(i, i_max, stride.get<0>()),
                                   xt::range(j, j_max, stride.get<1>()));
      std::cout << "clopped_padx: " << shape2string(clopped_padx.shape())
                << std::endl;
      auto clopped_col =
          xt::view(col, xt::all(), xt::all(), i, j, xt::all(), xt::all());
      std::cout << clopped_padx << std::endl;
      std::cout << "clopped_col: " << shape2string(clopped_col.shape())
                << std::endl;
      std::cout << clopped_col << std::endl;
*/
      xt::view(padx, xt::all(), xt::all(), xt::range(i, i_max, stride.get<0>()),
               xt::range(j, j_max, stride.get<1>())) +=
          xt::view(col, xt::all(), xt::all(), i, j, xt::all(), xt::all());
    }
  }
  // std::cout << "fin" << std::endl;
  return xt::eval(xt::view(
      padx, xt::all(), xt::all(),
      xt::range(padding.get<0>(), padx_shape[NCHW::H] - padding.get<0>()),
      xt::range(padding.get<1>(), padx_shape[NCHW::W] - padding.get<1>())));
}

namespace function {

class convolution_2d : virtual public traceable_function {
public:
  convolution_2d() : traceable_function{1} { set_name("convolution_2d"); }
  ~convolution_2d() = default;

  static tensor forward(const tensor &data, const tensor &weight,
                        const tensor &bias, exarray<2> stride = 1,
                        exarray<2> padding = 0, exarray<2> dilation = 1) {
    // std::cout << "x\n" << xt::mean(data.cdata(), {0, 1, 2}) << std::endl;
    // std::cout << "W\n" << xt::mean(weight.cdata()) << std::endl;
    // std::cout << "b\n" << xt::mean(bias.cdata()) << std::endl;

    assert(data.shape().size() == 4);
    assert(weight.shape().size() == 4);

    std::size_t N = data.shape()[NCHW::N];
    std::size_t W_f = weight.shape()[3];
    std::size_t H_f = weight.shape()[2];
    std::size_t C_in = weight.shape()[1];
    std::size_t C_out = weight.shape()[0];
    std::size_t H_out =
        (data.shape()[NCHW::H] + 2 * padding.get<0>() - H_f) / stride.get<0>() +
        1;
    std::size_t W_out =
        (data.shape()[NCHW::W] + 2 * padding.get<1>() - W_f) / stride.get<1>() +
        1;

    assert(bias.size() == 0 || bias.shape()[0] == C_out);

    auto col = im2col(data.cdata(), weight.shape(), stride, padding, dilation);

    // filter size for im2col is {C_out, C_in * H_f * W_f}.
    xt::xarray<tensor::value_type> filter = weight.cdata();
    filter.reshape({C_out, C_in * H_f * W_f});

    // shape is {N * H_out * W_out, C_out}
    auto dot = xt::linalg::dot(col, xt::transpose(filter));

    if (0 < bias.size()) {
      auto b = bias.cdata();
      dot += b;
    }

    dot.reshape({N, H_out, W_out, C_out});
    tensor::tensor_type result =
        xt::transpose(std::move(dot), {0, 3, 1, 2}); // {N, C_out, H_out, W_out}

    tensor output{std::move(result), util::requires_grad(data, weight, bias)};
    trace::register_node<convolution_2d>({data, weight, bias, stride.asTensor(),
                                          padding.asTensor(),
                                          dilation.asTensor()},
                                         output);

    return output;
  }

  static void backward(const std::vector<tensor> &outputs,
                       std::vector<tensor> &inputs) {
    // std::cout << "conv2d backward()" << std::endl;
    assert(outputs.size() == 1);
    assert(inputs.size() == 6);

    auto gy = outputs[0].cgrad(); // {N, C_out, (H_in + pad*2 - H_f)/stride +
                                  // 1, (W_in + pad*2 - W_f)/stride + 1}
    // std::cout << "gy\n" << xt::mean(gy) << std::endl;

    auto &data = inputs[0];
    auto &weight = inputs[1];
    auto &bias = inputs[2];
    auto stride = exarray<2>{inputs[3]};
    auto padding = exarray<2>{inputs[4]};
    auto dilation = exarray<2>{inputs[5]};

    auto &dx = data.grad(); // {N, C_in, H_in, W_in}
    // auto &dW = weight.grad(); // {C_out, C_in, H_f, W_f}

    std::size_t H_f = weight.shape()[2];
    std::size_t W_f = weight.shape()[3];

    // std::cout << shape2string(outputs[0].cdata().shape()) << std::endl;
    // std::cout << shape2string(outputs[0].cgrad().shape()) << std::endl;

    std::vector<std::size_t> y_shape = outputs[0].shape();
    assert(y_shape[NCHW::N] == data.shape()[NCHW::N]);

    std::size_t N = y_shape[NCHW::N];
    std::size_t C_out = y_shape[NCHW::C];
    std::size_t C_in = data.shape()[NCHW::C];

    gy = xt::transpose(gy, {0, 2, 3, 1});
    gy.reshape({-1, (int)C_out}); // {N * H_out * W_out, C_out}

    // db
    if (0 < bias.size() && bias.requires_grad()) {
      assert(y_shape[NCHW::C] == bias.shape()[0]);
      auto &db = bias.grad();
      db = xt::sum(gy, {0});
      // std::cout << "db\n" << db << std::endl << std::endl;
    }

    if (data.requires_grad() || weight.requires_grad()) {
      // {N * H_out * W_out, H_f * W_f * C_in}
      xt::xtensor<value_type, 2> col =
          im2col(data.data(), weight.shape(), stride, padding, dilation);
      // std::cout << "col shape: " << shape2string(col.shape()) << std::endl;
      // std::cout << "col\n" << xt::mean(col) << std::endl;
      if (weight.requires_grad()) {
        // dW

        // auto tcol = xt::transpose(col);
        // std::cout << "www shape: " << shape2string(www.shape()) << std::endl;
        // std::cout << xt::mean(www, {0}) << std::endl;
        /*
        auto www =
            xt::xtensor<float, 2>::from_shape({tcol.shape()[0], gy.shape()[1]});
        for (int i = 0; i < tcol.shape()[0]; i++) {
          for (int j = 0; j < gy.shape()[1]; j++) {
            auto c = xt::view(tcol, i, xt::all());
            auto r = xt::view(gy, xt::all(), j);
            xt::view(www, i, j) = xt::linalg::vdot(c, r);
          }
        }
        */

        auto &&dW =
            xt::linalg::dot(xt::transpose(col),
                            gy); // {Cols, N * H_out * W_out} x {N * H_out
                                 // * W_out, C_out} --> {Cols, C_out}
        // std::cout << "dW shape: " << shape2string(dW.shape()) << std::endl;
        // std::cout << "dW\n" << xt::mean(dW, {0}) << std::endl << std::endl;

        dW = xt::transpose(dW, {1, 0});

        assert(dW.shape()[0] == C_out);
        assert(dW.shape()[1] == C_in * H_f * W_f);

        dW.reshape({(int)C_out, (int)C_in, -1});
        dW.reshape({C_out, C_in, H_f, W_f});
        inputs[1].set_grad(std::move(dW));
      }
      if (data.requires_grad()) {
        xt::xarray<tensor::value_type> filter = weight.cdata();
        filter.reshape({(int)C_out, -1});

        // std::cout << "filter: " << filter << std::endl;
        // std::cout << "gy: " << gy << std::endl;
        auto dcol = xt::linalg::dot(
            gy, filter); // {N * H_out * W_out, C_out} x {C_out, Cols}
        // std::cout << "dcol: " << dcol << std::endl;

        dx = col2im(dcol, data.shape(), weight.shape(), stride, padding,
                    dilation);
        // std::cout << "dX\n" << xt::mean(dx) << std::endl << std::endl;
        // inputs[0].set_grad(dx);
      }
    }
  }
};
} // namespace function
} // namespace kuu

#endif // KUU_FUNCTIONS_CONVOLUTION_HPP