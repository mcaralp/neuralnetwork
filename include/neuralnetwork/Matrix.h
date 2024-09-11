
#ifndef NEURAL_NETWORK_MATRIX_H__
#define NEURAL_NETWORK_MATRIX_H__

#include "neuralnetwork/Vector.h"
#include <cstdint>
#include <ostream>

namespace nn
{
    template<typename T, uint32_t N, uint32_t M>
    class Matrix
    {
    public:
        static constexpr uint32_t Rows = N;
        static constexpr uint32_t Cols = M;

        Matrix()
        {
            std::memset(m_data, 0, Rows * Cols * sizeof(T));
        }

        Matrix(std::initializer_list<std::initializer_list<T>> list)
        {
            uint32_t rowLimit = std::min(Rows, static_cast<uint32_t>(list.size()));
            for (uint32_t i = 0; i < rowLimit; ++i)
            {
                const auto& l = *(std::data(list) + i);
                uint32_t colLimit = std::min(Cols, static_cast<uint32_t>(l.size()));
                std::memcpy(m_data + i * Cols, std::data(l), colLimit * sizeof(T));
                if (colLimit < Cols)
                {
                    std::memset(m_data + i * Cols + colLimit, 0, (Cols - colLimit) * sizeof(T));
                }
            }

            if (rowLimit < Rows)
            {
                std::memset(m_data + rowLimit * Cols, 0, (Rows - rowLimit) * Cols * sizeof(T));
            }

        }

        template<typename F>
        Matrix(F& f)
        {
            fill(f);
        }

        template<typename F>
        void fill(F& f)
        {
            for (uint32_t i = 0; i < Rows * Cols; ++i)
            {
                m_data[i] = f();
            }
        }

        const T* operator[](uint32_t row) const
        {
            return &m_data[row * Cols];
        }

        T* operator[](uint32_t row)
        {
            return &m_data[row * Cols];
        }

        const T& at(uint32_t row, uint32_t col) const
        {
            return m_data[row * Cols + col];
        }

        T& at(uint32_t row, uint32_t col)
        {
            return m_data[row * Cols + col];
        }

        const T& at(uint32_t index) const
        {
            return m_data[index];
        }

        T& at(uint32_t index)
        {
            return m_data[index];
        }

        Matrix<T, Rows, Cols> add(const Matrix<T, Rows, Cols>& rhs)
        {
            Matrix<T, Rows, Cols> result;
            for (uint32_t i = 0; i < Rows * Cols; ++i)
            {
                result.m_data[i] = m_data[i] + rhs.at(i);
            }
            return result;
        }

        template<typename F>
        Matrix<T, Rows, Cols> apply(const F& func)
        {
            Matrix<T, Rows, Cols> result;
            for (uint32_t i = 0; i < Rows * Cols; ++i)
            {
                result.m_data[i] = func(m_data[i]);
            }
            return result;
        }

        Vector<T, Rows> product(const Vector<T, Cols>& rhs)
        {
            Vector<T, Rows> result;
            for (uint32_t i = 0; i < Rows; ++i)
            {
                for (uint32_t j = 0; j < Cols; ++j)
                {
                    result.m_data[i] += m_data[i * Cols + j] * rhs.at(j);
                }
            }
            return result;
        }

        Matrix<T, Rows, Cols> hadamardProduct(const Matrix<T, Rows, Cols>& rhs)
        {
            Matrix<T, Rows, Cols> result;
            for (uint32_t i = 0; i < Rows * Cols; ++i)
            {
                result.m_data[i] = m_data[i] * rhs.at(i);
            }
            return result;
        }

    private:
        T m_data[Rows * Cols];

        template<typename U, uint32_t A, uint32_t B>
        friend class Matrix;
    };

    template<typename T, uint32_t N, uint32_t M>
    std::ostream& operator<<(std::ostream& os, const Matrix<T, N, M>& matrix)
    {
        for (uint32_t i = 0; i < N; ++i)
        {
            for (uint32_t j = 0; j < M; ++j)
            {
                os << matrix.at(i, j) << " ";
            }
            os << std::endl;
        }
        return os;
    }
}



#endif // NEURAL_NETWORK_MATRIX_H__
