#if 0
#ifndef FUNC_SIGMOID_H_
#define FUNC_SIGMOID_H_

#include <cassert>
#include <string>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "functions/math.hpp"

namespace kuu 
{
    namespace func
    {
        class Sigmoid
        {
        public:
            Sigmoid(
                std::string name = "sigmoid-activation"
                ): name_(name)
                {}
            template <class E>
            auto Forward(E&& x, const bool cash = true)
            {
                if (cash)
                {
                    x_ = x;
                }
                return math::sigmoid(x);
            }

            template <class E1, class E2, class E3>
            void Backword(E1&& error, E2&& weights, E3&& y)
            {
                if (y.empty())
                {
                    return;
                } 
                error_ = derivative(y) * xt::linalg::dot(error, weights);
            }

            std::string Name(void) const
            {
                return name_;
            }

        private:
            const std::string name_;
            xt::xarray<float> x_;
            xt::xarray<float> error_;

            template <class E>
            E& derivative(E&& x)
            {
                return x_ * (1 - x_);
            }
        };
    }
}
#endif // FUNC_SIGMOID_H_
#endif