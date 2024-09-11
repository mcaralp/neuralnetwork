
#ifndef NEURAL_NETWORK_SIGMOID_H__
#define NEURAL_NETWORK_SIGMOID_H__

#include <cmath>

namespace nn
{
    template<typename T>
    struct Sigmoid
    {
        using Type = T;

        T compute(const T& x) const
        {
            return 1 / (1 + std::exp(-x));
        }

        T derivative(const T& x) const
        {
            return compute(x) * (1 - compute(x));
        }

        T operator()(const T& x) const
        {
            return compute(x);
        }
    };
}

#endif // NEURAL_NETWORK_SIGMOID_H__
