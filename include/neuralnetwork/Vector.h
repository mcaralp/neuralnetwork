
#ifndef NEURAL_NETWORK_VECTOR_H__
#define NEURAL_NETWORK_VECTOR_H__

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <ostream>

namespace nn
{
    template<typename T, uint32_t N>
    class Vector
    {
    public:
        static constexpr uint32_t Size = N;

        Vector()
        {
            std::memset(m_data, 0, sizeof(m_data));
        }

        Vector(std::initializer_list<T> list)
        {
            uint32_t limit = std::min(Size, static_cast<uint32_t>(list.size()));
            std::memcpy(m_data, std::data(list), limit * sizeof(T));
            if (limit < Size)
            {
                std::memset(m_data + limit, 0, (Size - limit) * sizeof(T));
            }
        }

        template<typename F>
        Vector(F& f)
        {
            fill(f);
        }

        template<typename F>
        void fill(F& f)
        {
            for (uint32_t i = 0; i < Size; ++i)
            {
                m_data[i] = f();
            }
        }

        const T& operator[](uint32_t index) const
        {
            return m_data[index];
        }

        T& operator[](uint32_t index)
        {
            return m_data[index];
        }

        const T& at(uint32_t index) const
        {
            return m_data[index];
        }

        T& at(uint32_t index)
        {
            return m_data[index];
        }

        Vector<T, Size> add(const Vector<T, Size>& rhs)
        {
            Vector<T, Size> result;
            for (uint32_t i = 0; i < Size; ++i)
            {
                result.m_data[i] = m_data[i] + rhs.at(i);
            }
            return result;
        }

        template<typename F>
        Vector<T, Size> apply(const F& func)
        {
            Vector<T, Size> result;
            for (uint32_t i = 0; i < Size; ++i)
            {
                result.m_data[i] = func(m_data[i]);
            }
            return result;
        }

        T product(const Vector<T, Size>& rhs)
        {
            T result = 0;
            for (uint32_t i = 0; i < Size; ++i)
            {
                result += m_data[i] * rhs.at(i);
            }
            return result;
        }

        Vector<T, Size> hadamardProduct(const Vector<T, Size>& rhs)
        {
            Vector<T, Size> result;
            for (uint32_t i = 0; i < Size; ++i)
            {
                result.m_data[i] = m_data[i] * rhs.at(i);
            }
            return result;
        }

        T length() const
        {
            T result = 0;
            for (uint32_t i = 0; i < Size; ++i)
            {
                result += m_data[i] * m_data[i];
            }
            return std::sqrt(result);
        }

    private:
        T m_data[Size];

        template<typename U, uint32_t A, uint32_t B>
        friend class Matrix;
    };

    template<typename T, uint32_t N>
    std::ostream& operator<<(std::ostream& os, const Vector<T, N>& vec)
    {
        os << "[";
        for (uint32_t i = 0; i < N; ++i)
        {
            os << vec[i];
            if (i < N - 1)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
}

#endif // NEURAL_NETWORK_VECTOR_H__
