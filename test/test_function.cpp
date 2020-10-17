
#include "function.hpp"
#include "functions.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>
#include <string>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

TEST(FunctionTest, TestTraceableFunction) {
  kuu::traceable_function f{3};
  f.set_name("test");
  ASSERT_EQ(f.name(), "test");
  ASSERT_TRUE(f.id() != "");
  ASSERT_EQ(f.n_output(), 3);
}

TEST(FunctionTest, TestMSEForward) {
  kuu::tensor_type data0 = {1, 2};
  kuu::tensor_type data1 = {5, 6};
  kuu::tensor t0{data0};
  kuu::tensor t1{data1};
  kuu::tensor res = kuu::function::mean_squared_error::forward(t0, t1);
  ASSERT_EQ(res.data()(), 16);
}

TEST(FunctionTest, TestMSEBackward) {
  kuu::tensor_type x0 = {2, 3, 4};
  kuu::tensor_type x1 = {5, 6, 7};
  kuu::tensor_type y = {9};
  std::vector<kuu::tensor> out, in;
  out.emplace_back(y, true);
  out[0].set_grad(xt::ones_like(y));
  in.emplace_back(x0, true);
  in.emplace_back(x1, true);
  kuu::function::mean_squared_error::backward(out, in);

  kuu::tensor_type dx = {-2, -2, -2};
  ASSERT_EQ(in[0].grad(), dx);
  ASSERT_EQ(in[1].grad(), -1 * dx);
}

TEST(FunctionTest, TestLinearForward) {
  kuu::tensor_type data = {{2, 3}};
  kuu::tensor_type w = {{1, 2}, {3, 4}};
  kuu::tensor_type b = {5, 6};
  kuu::tensor t0{data, true};
  kuu::tensor t1{w, true};
  kuu::tensor t2{b, true};
  kuu::tensor res = kuu::function::linear::forward(t0, t1, t2);
  kuu::tensor_type ans0 = {{11 + 5, 16 + 6}};
  TENSOR_CLONE_EQ(res, kuu::tensor{ans0});
  ASSERT_EQ(res.data(), ans0);
  res = kuu::function::linear::forward(t0, t1);
  kuu::tensor_type ans1 = {{11, 16}};
  TENSOR_CLONE_EQ(res, kuu::tensor{ans1});
}

TEST(FunctionTest, TestLinearBackward) {
  kuu::tensor_type x = {{2, 3}};         // 1 x 2
  kuu::tensor_type w = {{1, 2}, {3, 4}}; // 2 x 2
  kuu::tensor_type b = {5, 6};           // 2
  kuu::tensor t0{x, true};
  kuu::tensor t1{w, true};
  kuu::tensor t2{b, true};
  kuu::tensor_type y = {{11 + 5, 16 + 6}}; // 1 x 2
  std::vector<kuu::tensor> out, in;
  out.emplace_back(y, true);
  kuu::tensor_type gy = xt::ones_like(y);
  out[0].set_grad(gy);

  in.emplace_back(x, true);
  in.emplace_back(w, true);
  in.emplace_back(b, true);

  kuu::function::linear::backward(out, in);
  kuu::tensor_type gx = {{3, 7}};
  kuu::tensor_type gw = {{2, 2}, {3, 3}};
  kuu::tensor_type gb = {1, 1};
  ASSERT_EQ(in[0].grad(), gx);
  ASSERT_EQ(in[1].grad(), gw);
  ASSERT_EQ(in[2].grad(), gb);
}

TEST(FunctionTest, TestReluForward) {
  kuu::tensor_type x = {{-2, 3}, {3, -5}};
  kuu::tensor in{x};
  kuu::tensor_type t = {{0, 3}, {3, 0}};
  auto out = kuu::function::relu::forward(in);
  TENSOR_CLONE_EQ(out, kuu::tensor{t});
}

TEST(FunctionTest, TestReluBackward) {
  kuu::tensor_type x = {{-2, 3}, {3, -5}};
  kuu::tensor_type y = {{0, 3}, {3, 0}};

  std::vector<kuu::tensor> in;
  in.emplace_back(x, true);
  std::vector<kuu::tensor> out;
  out.emplace_back(y, true);
  kuu::tensor_type gy = xt::ones_like(y);
  out[0].set_grad(gy);

  kuu::tensor_type gx = {{0, 1}, {1, 0}};
  kuu::function::relu::backward(out, in);
  ASSERT_EQ(in[0].grad(), gx);
  ASSERT_EQ(in[0].data(), x);
}

TEST(FunctionTest, TestConv2dForward) {
  kuu::tensor_type x = xt::ones<kuu::value_type>({2, 1, 5, 5});
  kuu::tensor_type w = xt::ones<kuu::value_type>({1, 1, 3, 3});
  kuu::tensor_type b = {2};

  std::vector<double> v{1, 2, 3,     4, 5, 2, 1,  3, 1,  1,  0, 1,
                        0, 1, 1 - 1, 1, 2, 1, -1, 0, -1, -2, 1, 2};
  std::vector<std::size_t> shape = {5, 5};
  auto reshape = xt::adapt(v, shape);
  xt::view(x, 0, 0, xt::all(), xt::all()) = reshape;

  // std::cout << x << std::endl;

  kuu::tensor data{std::move(x)};
  kuu::tensor weight{std::move(w)};
  kuu::tensor bias{std::move(b)};
  auto result = kuu::function::convolution_2d::forward(data, weight, bias);

  xt::xarray<kuu::value_type> gt0 = {
      {{15., 18., 20.}, {13., 11., 8.}, {5., 7., 6.}}};
  xt::xarray<kuu::value_type> gt1 = xt::ones<kuu::value_type>({1, 3, 3}) * 11;

  std::vector<std::size_t> res_shape = {2, 1, 3, 3};
  ASSERT_EQ(result.shape(), res_shape);
  xt::xarray<kuu::value_type> y0 =
      xt::view(result.data(), 0, xt::all(), xt::all(), xt::all());
  xt::xarray<kuu::value_type> y1 =
      xt::view(result.data(), 1, xt::all(), xt::all(), xt::all());
  ASSERT_EQ(y0.storage(), gt0.storage());
  ASSERT_EQ(y1.storage(), gt1.storage());
}

TEST(FunctionTest, TestConv2dBackward) {
  kuu::tensor_type x = {{{{-0.5800, -2.0131, 0.5050, -0.1517, -0.5042},
                          {-1.2957, -0.7817, 2.5573, 0.3447, -0.4917},
                          {-0.0191, -0.7280, 0.6636, 0.2440, 0.2484},
                          {1.3338, 0.0537, -1.0760, -1.2439, 1.3036},
                          {-0.0836, 0.2876, -1.0337, 1.1563, 0.5683}},
                         {{0.2105, 0.1417, 0.6034, 1.0286, 0.1786},
                          {-0.8337, -0.6166, -2.3295, 1.4396, -1.3064},
                          {-1.0435, -0.7919, 0.7596, 1.5828, -0.2398},
                          {-0.3571, 0.1924, -0.8016, 1.0583, 0.8047},
                          {-0.2977, -0.8465, 0.1196, 1.7097, 0.1296}},
                         {{-1.8815, 0.7994, 0.7929, -0.0154, -0.6016},
                          {0.1303, 0.9617, -2.3317, 0.6128, -0.2475},
                          {-1.2643, -2.1434, -0.5084, -0.1740, -0.0424},
                          {-0.5184, -0.7487, -1.3985, 2.7875, -0.2291},
                          {-1.6471, 0.1465, -0.6131, -0.4413, 1.4151}}},
                        {{{-0.1553, -0.8518, 0.5705, 0.5591, 0.1694},
                          {1.2458, 1.2583, -1.8627, 0.0538, 0.2039},
                          {0.3159, 2.4130, -1.3934, 0.7046, 1.1963},
                          {-0.8285, 0.6577, 1.4914, 0.1332, -0.0447},
                          {0.3258, -0.1086, -1.4373, -0.1453, -0.3339}},
                         {{-0.1537, -0.0486, -0.9712, -0.2864, 0.1982},
                          {-0.9934, -0.8513, 0.5067, -0.8808, 2.2660},
                          {0.5011, -2.0296, -1.1131, -0.3094, -1.0652},
                          {-0.5914, -0.3142, 2.0109, -1.6630, 0.1396},
                          {-1.4525, 1.7656, 0.8046, -1.0552, 0.8080}},
                         {{-0.2236, 0.5631, 0.7518, -0.4415, 0.2143},
                          {-0.2690, 0.5780, -0.8985, -0.7452, 0.2188},
                          {0.3224, -0.1224, -0.6641, 0.4140, 0.8583},
                          {-0.3468, -0.5239, 1.2989, 0.8407, -1.8906},
                          {0.8129, 0.5564, 1.0924, 0.5543, -0.5296}}},
                        {{{0.9416, -0.1513, -1.1793, -0.9617, 0.4835},
                          {-0.1252, -0.2701, -1.4790, -0.5966, 0.1694},
                          {1.7271, 0.9288, -1.0472, 1.6885, -1.4024},
                          {1.0692, 0.1620, 1.2930, -0.4458, -1.1093},
                          {1.5273, 0.1311, -0.6233, -0.7806, 0.1410}},
                         {{1.0321, -0.5986, 0.5422, 1.4088, 1.4155},
                          {1.0227, -1.0880, 1.8697, -0.2098, 0.4862},
                          {0.4060, 0.2446, 0.6386, 1.4550, -1.5625},
                          {-0.0117, -1.3592, -1.0911, -0.0994, 0.6834},
                          {1.0034, -0.5520, 1.1793, -1.0357, 0.0957}},
                         {{-1.1521, 0.7243, 0.4063, 0.7334, 0.8487},
                          {-1.0713, 0.5993, 0.2454, -1.1199, 1.2792},
                          {0.2879, 1.1261, 0.2701, -0.0785, 1.3430},
                          {0.1259, -0.7975, -1.3148, 0.6435, 1.1924},
                          {-0.1881, -0.7418, -1.7321, -0.3112, 0.0493}}}};
  kuu::tensor_type w = {{{{1.1060, 0.8169, 1.5466},
                          {0.9841, -0.1308, 0.0762},
                          {1.2691, 0.5821, -1.7505}},
                         {{1.4728, -0.7603, -0.0263},
                          {0.8511, -0.9544, 0.6193},
                          {-0.0643, -0.3419, 1.3164}},
                         {{-0.2978, 0.4882, 0.9036},
                          {-0.3176, 0.3337, -2.3555},
                          {1.2437, 0.9916, 0.4611}}},
                        {{{0.6211, 0.4507, -0.4680},
                          {2.2555, -0.0395, 1.1575},
                          {0.4551, 0.0346, -0.0644}},
                         {{-1.0600, -0.4562, 1.9603},
                          {0.4724, -2.6464, 0.0712},
                          {-0.1013, -1.0589, -0.0614}},
                         {{-0.4902, 1.0098, -0.8233},
                          {-1.7615, -0.4038, 0.1686},
                          {1.6232, -0.6515, 0.7545}}}};
  kuu::tensor_type b = {2, 3};

  // std::cout << "x: " << x << std::endl;
  // std::cout << "W: " << w << std::endl;

  std::vector<kuu::tensor> in;
  in.emplace_back(x, true);
  in.emplace_back(w, true);
  in.emplace_back(std::move(b), true);
  in.emplace_back(kuu::exarray<2>{1, 1}.asTensor()); // stride
  in.emplace_back(kuu::exarray<2>{0, 0}.asTensor()); // padding
  in.emplace_back(kuu::exarray<2>{1, 1}.asTensor());

  kuu::tensor_type y = {{{{1.3747, -3.5450, -0.4147},
                          {3.3360, 5.6708, -2.2447},
                          {3.9256, -5.1759, 0.5878}},
                         {{3.3796, 3.4982, 4.6261},
                          {4.6398, 7.6388, -0.8873},
                          {3.9116, 10.5531, -3.1125}}},
                        {{{9.8809, 4.3699, -2.4391},
                          {2.3919, -0.7132, 1.6490},
                          {6.3314, -2.0303, 13.5496}},
                         {{5.6916, 6.5881, 3.8882},
                          {13.5822, 7.4002, 7.5342},
                          {5.2279, 0.7262, 8.2450}}},
                        {{{13.8426, -0.8051, -0.9939},
                          {1.1011, -2.8521, 1.0085},
                          {11.6988, -0.9210, -3.8460}},
                         {{8.9538, -1.5569, 1.2656},
                          {10.4670, 5.2178, -10.2692},
                          {13.7805, 7.1316, -0.1706}}}};

  std::vector<kuu::tensor> out;
  out.emplace_back(y, true);
  kuu::tensor_type gy = xt::ones_like(y);
  out[0].set_grad(gy);

  kuu::function::convolution_2d::backward(out, in);

  auto &db = in[2].grad();
  auto &dW = in[1].grad();
  auto &dx = in[0].grad();

  xt::xarray<float> expected_db = {27., 27.}; // xt::sum(gy, {0, 2, 3});
  CLOSE_ALL(db, expected_db);

  xt::xtensor<float, 4> &&pytorch_dx = {
      {{{1.7271e+00, 2.9947e+00, 4.0733e+00, 2.3463e+00, 1.0786e+00},
        {4.9667e+00, 6.0640e+00, 8.3763e+00, 3.4096e+00, 2.3123e+00},
        {6.6909e+00, 8.4049e+00, 8.9023e+00, 2.2114e+00, 4.9742e-01},
        {4.9638e+00, 5.4102e+00, 4.8290e+00, -1.3489e-01, -5.8121e-01},
        {1.7242e+00, 2.3409e+00, 5.2598e-01, -1.1982e+00, -1.8149e+00}},
       {{4.1281e-01, -8.0368e-01, 1.1303e+00, 7.1750e-01, 1.9340e+00},
        {1.7363e+00, -3.0810e+00, -4.5655e-01, -2.1928e+00, 2.6245e+00},
        {1.5707e+00, -4.6475e+00, -7.6795e-01, -2.3386e+00, 3.8795e+00},
        {1.1579e+00, -3.8438e+00, -1.8983e+00, -3.0561e+00, 1.9455e+00},
        {-1.6561e-01, -1.5664e+00, -3.1141e-01, -1.4580e-01, 1.2550e+00}},
       {{-7.8799e-01, 7.1000e-01, 7.9036e-01, 1.5783e+00, 8.0368e-02},
        {-2.8670e+00, -1.4392e+00, -3.5457e+00, -6.7870e-01, -2.1066e+00},
        {-1.6546e-04, 1.7678e+00, 8.7675e-01, 8.7691e-01, -8.9103e-01},
        {7.8782e-01, 1.0578e+00, 8.6383e-02, -7.0144e-01, -9.7140e-01},
        {2.8669e+00, 3.2069e+00, 4.4225e+00, 1.5556e+00, 1.2156e+00}}},
      {{{1.7271e+00, 2.9947e+00, 4.0733e+00, 2.3463e+00, 1.0786e+00},
        {4.9667e+00, 6.0640e+00, 8.3763e+00, 3.4096e+00, 2.3123e+00},
        {6.6909e+00, 8.4049e+00, 8.9023e+00, 2.2114e+00, 4.9742e-01},
        {4.9638e+00, 5.4102e+00, 4.8290e+00, -1.3489e-01, -5.8121e-01},
        {1.7242e+00, 2.3409e+00, 5.2598e-01, -1.1982e+00, -1.8149e+00}},
       {{4.1281e-01, -8.0368e-01, 1.1303e+00, 7.1750e-01, 1.9340e+00},
        {1.7363e+00, -3.0810e+00, -4.5655e-01, -2.1928e+00, 2.6245e+00},
        {1.5707e+00, -4.6475e+00, -7.6795e-01, -2.3386e+00, 3.8795e+00},
        {1.1579e+00, -3.8438e+00, -1.8983e+00, -3.0561e+00, 1.9455e+00},
        {-1.6561e-01, -1.5664e+00, -3.1141e-01, -1.4580e-01, 1.2550e+00}},
       {{-7.8799e-01, 7.1000e-01, 7.9036e-01, 1.5783e+00, 8.0368e-02},
        {-2.8670e+00, -1.4392e+00, -3.5457e+00, -6.7870e-01, -2.1066e+00},
        {-1.6546e-04, 1.7678e+00, 8.7675e-01, 8.7691e-01, -8.9103e-01},
        {7.8782e-01, 1.0578e+00, 8.6383e-02, -7.0144e-01, -9.7140e-01},
        {2.8669e+00, 3.2069e+00, 4.4225e+00, 1.5556e+00, 1.2156e+00}}},
      {{{1.7271e+00, 2.9947e+00, 4.0733e+00, 2.3463e+00, 1.0786e+00},
        {4.9667e+00, 6.0640e+00, 8.3763e+00, 3.4096e+00, 2.3123e+00},
        {6.6909e+00, 8.4049e+00, 8.9023e+00, 2.2114e+00, 4.9742e-01},
        {4.9638e+00, 5.4102e+00, 4.8290e+00, -1.3489e-01, -5.8121e-01},
        {1.7242e+00, 2.3409e+00, 5.2598e-01, -1.1982e+00, -1.8149e+00}},
       {{4.1281e-01, -8.0368e-01, 1.1303e+00, 7.1750e-01, 1.9340e+00},
        {1.7363e+00, -3.0810e+00, -4.5655e-01, -2.1928e+00, 2.6245e+00},
        {1.5707e+00, -4.6475e+00, -7.6795e-01, -2.3386e+00, 3.8795e+00},
        {1.1579e+00, -3.8438e+00, -1.8983e+00, -3.0561e+00, 1.9455e+00},
        {-1.6561e-01, -1.5664e+00, -3.1141e-01, -1.4580e-01, 1.2550e+00}},
       {{-7.8799e-01, 7.1000e-01, 7.9036e-01, 1.5783e+00, 8.0368e-02},
        {-2.8670e+00, -1.4392e+00, -3.5457e+00, -6.7870e-01, -2.1066e+00},
        {-1.6546e-04, 1.7678e+00, 8.7675e-01, 8.7691e-01, -8.9103e-01},
        {7.8782e-01, 1.0578e+00, 8.6383e-02, -7.0144e-01, -9.7140e-01},
        {2.8669e+00, 3.2069e+00, 4.4225e+00, 1.5556e+00, 1.2156e+00}}}};
  xt::xtensor<float, 4> &&expected_dx = xt::zeros_like(dW);
  xt::xtensor<float, 4> &&pytorch_dw = {{{{-0.8059, -0.9762, -0.7080},
                                          {6.2640, 3.7235, 0.1031},
                                          {6.0021, 1.9454, -1.2847}},
                                         {{-4.9839, 0.0965, 6.1053},
                                          {-8.0646, -3.7903, 3.0293},
                                          {-3.0273, 0.4592, 3.9434}},
                                         {{-3.9715, 0.3354, 1.1203},
                                          {-8.9757, -3.1917, 0.3614},
                                          {-9.2337, -2.5832, 2.8320}}},
                                        {{{-0.8059, -0.9762, -0.7080},
                                          {6.2640, 3.7235, 0.1031},
                                          {6.0021, 1.9454, -1.2847}},
                                         {{-4.9839, 0.0965, 6.1053},
                                          {-8.0646, -3.7903, 3.0293},
                                          {-3.0273, 0.4592, 3.9434}},
                                         {{-3.9715, 0.3354, 1.1203},
                                          {-8.9757, -3.1917, 0.3614},
                                          {-9.2337, -2.5832, 2.8320}}}};
  xt::xtensor<float, 4> &&expected_dw = xt::zeros_like(dx);

  /*
    std::size_t fh = w.shape()[2];
    std::size_t fw = w.shape()[3];

    for (std::size_t n = 0; n < x.shape()[0]; n++) {
        for (std::size_t i = 0; i < y.shape()[2]; i++) {
          for (std::size_t j = 0; j < y.shape()[3]; j++) {
            std::cout << "( " << n << ", " << i << ", " << j << " )" <<
    std::endl; xt::view(expected_dx, n, c, xt::range(i, i + fh), xt::range(j, j
    + fw)) += xt::view(w, 0, 0, xt::all(), xt::all()) * xt::view(gy, n, 0, i,
    j); expected_dw += xt::view(x, n, 0, xt::range(i, i + fh), xt::range(j, j +
    fw)) * xt::view(gy, n, 0, i, j);
          }
        }
    } */

  // std::cout << "dx: " << dx << std::endl;
  // std::cout << "expected_dx: " << expected_dx << std::endl;
  CLOSE_ALL(dW, pytorch_dw, 0.001);
  CLOSE_ALL(dx, pytorch_dx, 0.001);
  // CLOSE_ALL(dW, expected_dw);
  // CLOSE_ALL(dx, expected_dx);
}

TEST(FunctionTest, TestIm2Col) {
  xt::xarray<int> im = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                        10, 11, 12, 13, 14, 15, 16, 17, 18};
  im.reshape({1, 2, 3, 3});
  // std::cout << "im\n" << im << std::endl;
  auto col = kuu::im2col(im, std::vector<int>{2, 2, 3, 3}, 3, 3, 1);
  // std::cout << "col\n" << col << std::endl;

  xt::xarray<int> col4 = xt::eval(xt::view(col, 4, xt::all()));
  // std::cout << col4 << std::endl;
  ASSERT_EQ(col4.shape()[0], 18);
  ASSERT_EQ(col4.storage(), im.storage());

  xt::xarray<int> other =
      xt::concatenate(xt::xtuple(xt::view(col, xt::range(0, 4), xt::all()),
                                 xt::view(col, xt::range(5, 9), xt::all())));
  // std::cout << other << std::endl;
  xt::xarray<int> zero = xt::zeros<int>({1 * 3 * 3 - 1, 2 * 3 * 3});
  ASSERT_EQ(other.shape(), zero.shape());
  ASSERT_EQ(other.storage(), zero.storage());
}

TEST(FunctionTest, TestCol2Im) {
  xt::xarray<float> col = {
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
       17., 18.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}};

  xt::xarray<float> im = kuu::col2im(col, std::vector<int>{1, 2, 3, 3},
                                     std::vector<int>{2, 2, 3, 3}, 3, 3, 1);

  // std::cout << "im_shape: " << kuu::shape2string(im.shape()) << std::endl;
  // std::cout << "im: " << im << std::endl;
  xt::xarray<float> im_gt = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};

  im_gt.reshape({1, 2, 3, 3});
  // ASSERT_EQ(im.shape(), im_gt.shape());
  CLOSE_ALL(im, im_gt);
}

TEST(FunctionTest, TestSoftmaxCrossEntropyForward) {
  xt::xarray<float> x = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}; // {3, 3}

  xt::xarray<float> t0 = {1, 0, 2};
  xt::xarray<float> t1 = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

  kuu::tensor input{x};

  kuu::tensor label{t0};
  kuu::tensor softlabel{t1};

  kuu::tensor ans0 =
      kuu::function::softmax_cross_entropy::forward(input, label);
  kuu::tensor ans1 =
      kuu::function::softmax_cross_entropy::forward(input, softlabel);

  TENSOR_CLONE_EQ(ans0, ans1);
  EXPECT_NEAR(ans0.data()(), 1.0986, 0.0001);
}

TEST(FunctionTest, TestSoftmaxCrossEntropyBackward) {
  xt::xarray<float> x = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}; // {3, 3}

  xt::xarray<float> t0 = {1, 0, 2};
  xt::xarray<float> t1 = {{0, 1, 0}, {1, 0, 0}, {0, 0, 1}};

  std::vector<kuu::tensor> out;
  out.emplace_back(1.0986, true);
  out[0].set_grad(xt::ones_like(out[0].data()));
  std::vector<kuu::tensor> in;
  in.emplace_back(x, true);
  in.emplace_back(t0, true);
  kuu::function::softmax_cross_entropy::backward(out, in);

  auto scores = kuu::math::log_softmax(x, 1);
  std::cout << "scores: " << scores << std::endl;
  xt::xarray<float> torch_scores = {{-1.0986, -1.0986, -1.0986},
                                    {-1.0986, -1.0986, -1.0986},
                                    {-1.0986, -1.0986, -1.0986}};
  CLOSE_ALL(scores, torch_scores, 0.001);

  xt::xtensor<float, 2> torch_gx = {{0.1111, -0.2222, 0.1111},
                                    {-0.2222, 0.1111, 0.1111},
                                    {0.1111, 0.1111, -0.2222}};
  CLOSE_ALL(in[0].grad(), torch_gx, 0.001);
}

TEST(FunctionTest, TestBatchNorm1dForward) {
  xt::xarray<float> x = {{1, 0, 3}, {-1, 2, 0}, {0, -2, -3}}; // {3, 3}
  xt::xarray<float> gamma = {5, 6, 7};
  xt::xarray<float> beta = {1, 2, 3};
  xt::xarray<float> running_mean = {3, 2, 1};
  xt::xarray<float> running_var = {4, 4, 4};
  double eps = 1e-5;

  kuu::tensor input{x};
  kuu::tensor w{gamma};
  kuu::tensor b{beta};
  kuu::tensor mean{running_mean};
  kuu::tensor var{running_var};

  xt::xarray<float> y =
      (x - running_mean) * gamma / xt::sqrt(running_var + eps) + beta;
  auto out0 = kuu::function::batchnorm::forward(input, w, b, mean, var, eps,
                                                0.1, false);

  CLOSE_ALL(out0.data(), y);
  EXPECT_EQ(out0.data().shape(), y.shape());

  auto batch_mean = xt::mean(x, {0});
  auto batch_var = xt::variance(x, {0});

  auto out1 =
      kuu::function::batchnorm::forward(input, w, b, mean, var, eps, 0.1, true);
  y = (x - batch_mean) * gamma / xt::sqrt(batch_var + eps) + beta;

  CLOSE_ALL(out1.data(), y);
  EXPECT_EQ(out1.data().shape(), y.shape());
}

TEST(FunctionTest, TestBatchNormNdForward) {
  xt::xarray<float> x = {{{{1, 1}}, {{1, 1}}, {{1, 1}}},
                         {{{2, 2}}, {{2, 2}}, {{2, 2}}},
                         {{{3, 3}}, {{3, 3}}, {{3, 3}}}}; // {3, 3, 1, 2}
  xt::xarray<float> gamma = {5, 6, 7};
  xt::xarray<float> beta = {1, 2, 3};
  xt::xarray<float> running_mean = {1, 2, 3};
  xt::xarray<float> running_var = {3, 2, 1};
  double eps = 1e-5;
  xt::xtensor<int, 1> crange = xt::arange(x.shape()[1]);
  xt::xarray<float> batch_mean = xt::mean(x, {0});
  xt::xarray<float> batch_var = xt::variance(x, {0});

  auto x_shape = x.shape();
  x.reshape({3, 3, -1});

  xt::xarray<float> y0{x.shape()};
  std::for_each(crange.begin(), crange.end(),
                [&y0, &gamma, &beta, &batch_mean, &batch_var, &x, eps](auto i) {
                  xt::view(y0, xt::all(), i, xt::all()) =
                      (xt::view(x, xt::all(), i, xt::all()) - batch_mean(i)) *
                          gamma(i) / std::sqrt(batch_var(i) + eps) +
                      beta(i);
                });

  xt::xarray<float> y1 = xt::xarray<float>::from_shape(x.shape());
  std::for_each(
      crange.begin(), crange.end(),
      [&y1, &gamma, &beta, &running_mean, &running_var, &x, eps](auto i) {
        xt::view(y1, xt::all(), i, xt::all()) =
            (xt::view(x, xt::all(), i, xt::all()) - running_mean(i)) *
                gamma(i) / std::sqrt(running_var(i) + eps) +
            beta(i);
      });

  y0.reshape(x_shape);
  y1.reshape(x_shape);
  x.reshape(x_shape);

  kuu::tensor input{x};
  kuu::tensor w{gamma};
  kuu::tensor b{beta};
  kuu::tensor mean{running_mean};
  kuu::tensor var{running_var};

  auto out0 =
      kuu::function::batchnorm::forward(input, w, b, mean, var, eps, 0.1, true);
  auto out1 = kuu::function::batchnorm::forward(input, w, b, mean, var, eps,
                                                0.1, false);

  running_mean.reshape({1, 3, 1});
  running_var.reshape({1, 3, 1});
  batch_mean.reshape({1, 3, 1});
  batch_var.reshape({1, 3, 1});
  x.reshape({3, 3, -1});
  xt::xarray<float> y2 = (x - running_mean) / xt::sqrt(running_var + eps);
  y2.reshape(x_shape);

  std::cout << "for_each: " << y1 << std::endl;
  std::cout << "reshape and *: " << y2 << std::endl;
  std::cout << "bn: " << out1.data() << std::endl;

  EXPECT_EQ(out0.shape(), x_shape);
  EXPECT_EQ(out1.shape(), x_shape);
  CLOSE_ALL(out0.data(), y0);
  CLOSE_ALL(out1.data(), y1);
}