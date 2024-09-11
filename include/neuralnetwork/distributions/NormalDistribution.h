
#ifndef NEURAL_NETWORK_NORMAL_DISTRIBUTION_H__
#define NEURAL_NETWORK_NORMAL_DISTRIBUTION_H__

#include <random>

namespace nn
{
    template<typename T>
    class NormalDistribution
    {
    public:
        NormalDistribution()
            : m_generator(m_device())
        {
        }

        T operator()()
        {
            return m_distribution(m_generator);
        }

    private:
        std::random_device m_device;
        std::mt19937 m_generator;
        std::normal_distribution<T> m_distribution;
    };
}

#endif // NEURAL_NETWORK_NORMAL_DISTRIBUTION_H__
