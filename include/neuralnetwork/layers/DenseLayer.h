
#ifndef NEURAL_NETWORK_DENSE_LAYER_H__
#define NEURAL_NETWORK_DENSE_LAYER_H__

#include "neuralnetwork/Matrix.h"

#include <cstdint>
#include <type_traits>

namespace nn
{
    template<typename T, typename F, uint32_t I, uint32_t O>
    class DenseLayer
    {
    public:
        static_assert(std::is_same_v<T, typename F::Type>, "Invalid type");

        using Type = T;
        static constexpr uint32_t Input = I;
        static constexpr uint32_t Output = O;

        DenseLayer()
        {
        }

        template<typename F1>
        void fill(F1& f)
        {
            m_weights.fill(f);
            m_biases.fill(f);
        }

        Vector<T, Output> forward(const Vector<T, Input>& input)
        {
            //  activation(weights * input + biases)
            return m_weights.product(input).add(m_biases).apply(m_activation);
        }

        const Matrix<T, Output, Input>& weights() const
        {
            return m_weights;
        }

        const Vector<T, Output>& biases() const
        {
            return m_biases;
        }
    private:
        Matrix<T, Output, Input> m_weights;
        Vector<T, Output> m_biases;
        F m_activation;
    };
}

#endif // NEURAL_NETWORK_DENSE_LAYER_H__
