
#ifndef NEURAL_NETWORK_RELU_H__
#define NEURAL_NETWORK_RELU_H__

namespace nn
{
    template<typename T>
    struct ReLU
    {
        using Type = T;

        T compute(const T& x) const
        {
            return x > 0 ? x : 0;
        }

        T derivative(const T& x) const
        {
            return x > 0 ? 1 : 0;
        }

        T operator()(const T& x) const
        {
            return compute(x);
        }
    };
}

#endif // NEURAL_NETWORK_DENSE_LAYER_H__
