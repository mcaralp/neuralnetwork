
#include "neuralnetwork/Model.h"
#include "neuralnetwork/activations/Sigmoid.h"
#include "neuralnetwork/layers/DenseLayer.h"
#include "neuralnetwork/distributions/NormalDistribution.h"
#include "neuralnetwork/dataset/MNIST.h"

#include <iostream>

int main()
{
    using namespace nn;

    MNIST mnist;

    std::cout << mnist.trainLabel()[0] << std::endl;

    using FirstLayer = DenseLayer<double, Sigmoid<double>, 784, 100>;
    using SecondLayer = DenseLayer<double, Sigmoid<double>, 100, 10>;
    using Model = Model<FirstLayer, SecondLayer>;

    NormalDistribution<double> normalDistribution;
    Model network;
    network.fill(normalDistribution);

    Vector<double, 784> input  = {1, 2};
    Vector<double, 10> output = network.forward(input);

    return 0;
}
