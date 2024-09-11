
#ifndef NEURAL_NETWORK_PRODUCT_H__
#define NEURAL_NETWORK_PRODUCT_H__

namespace nn
{
    template<typename T, T Value = 1>
    struct Product
    {
        using Type = T;

        T compute(const T& x) const
        {
            return x * Value;
        }

        T derivative(const T& x) const
        {
            return Value;
        }

        T operator()(const T& x) const
        {
            return compute(x);
        }
    };
}

#endif // NEURAL_NETWORK_DENSE_LAYER_H__
