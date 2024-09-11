
#ifndef NEURAL_NETWORK_MNIST_H__
#define NEURAL_NETWORK_MNIST_H__

#include "neuralnetwork/Vector.h"
 
extern const uint8_t trainImagesMNIST[];
extern const uint8_t trainLabelsMNIST[];
extern const uint8_t testImagesMNIST[];
extern const uint8_t testLabelsMNIST[];

namespace nn
{
    class MNIST
    {
    public:
        using Image = Vector<double, 784>;
        using Result = Vector<double, 10>;

        MNIST()
        {
            load();
        }

        ~MNIST()
        {
            delete[] m_trainImage;
            delete[] m_trainLabel;
            delete[] m_testImage;
            delete[] m_testLabel;
        }

        void load()
        {
            uint32_t trainImagesOffset = 16;
            uint32_t trainLabelsOffset = 8;
            uint32_t testImagesOffset = 16;
            uint32_t testLabelsOffset = 8;

            m_trainImage = new Image[60000];
            m_trainLabel = new Result[60000];
            m_testImage = new Image[10000];
            m_testLabel = new Result[10000];

            for (int i = 0; i < 60000; ++i)
            {
                for (int j = 0; j < 784; ++j)
                {
                    m_trainImage[i][j] = trainImagesMNIST[trainImagesOffset + i * 784 + j];
                }
                m_trainLabel[i][trainLabelsMNIST[trainLabelsOffset + i]] = 1;
            }

            for (int i = 0; i < 10000; ++i)
            {
                for (int j = 0; j < 784; ++j)
                {
                    m_testImage[i][j] = testImagesMNIST[testImagesOffset + i * 784 + j];
                }
                m_testLabel[i][testLabelsMNIST[testLabelsOffset + i]] = 1;
            }
        }

        const Image* trainImage() const
        {
            return m_trainImage;
        }

        const Image* testImage() const
        {
            return m_testImage;
        }

        const Result* trainLabel() const
        {
            return m_trainLabel;
        }

        const Result* testLabel() const
        {
            return m_testLabel;
        }

    private:

        Image* m_trainImage;
        Image* m_testImage;

        Result* m_trainLabel;
        Result* m_testLabel;
    };
}

#endif // NEURAL_NETWORK_MNIST_H__
