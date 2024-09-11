
#ifndef NEURAL_NETWORK_SEQUENCE_DISTRIBUTION_H__
#define NEURAL_NETWORK_SEQUENCE_DISTRIBUTION_H__

#include <random>

namespace nn
{
    template<typename T>
    class SequenceDistribution
    {
    public:
        SequenceDistribution(T start, T step)
            : m_start(start), m_step(step)
        {
        }

        T operator()()
        {
            T value = m_start;
            m_start += m_step;
            return value;
        }

    private:
        T m_start;
        T m_step;
    };
}

#endif // NEURAL_NETWORK_NORMAL_DISTRIBUTION_H__
