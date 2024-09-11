
#include "neuralnetwork/activations/Product.h"
#include "neuralnetwork/layers/DenseLayer.h"
#include "neuralnetwork/distributions/SequenceDistribution.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE( "Vectors" ) {

    nn::Vector<int32_t, 3> v1 = {1, 2, 3};

    SECTION("Initialization") {
        REQUIRE( v1[0] == 1 );
        REQUIRE( v1[1] == 2 );
        REQUIRE( v1[2] == 3 );
    }

    SECTION("Partial initialization") {
        nn::Vector<int32_t, 3> v2 = {1, 2};
        REQUIRE( v2[0] == 1 );
        REQUIRE( v2[1] == 2 );
        REQUIRE( v2[2] == 0 );
    }

    SECTION("Fill") {
        nn::SequenceDistribution<int32_t> distribution(10, 2);
        v1.fill(distribution);
        REQUIRE( v1[0] == 10 );
        REQUIRE( v1[1] == 12 );
        REQUIRE( v1[2] == 14 );
    }

    SECTION ("Addition") {
        nn::Vector<int32_t, 3> v2 = {4, 5, 6};
        auto v3 = v1.add(v2);
        REQUIRE( v3[0] == 5 );
        REQUIRE( v3[1] == 7 );
        REQUIRE( v3[2] == 9 );
    }

    SECTION("Product") {
        nn::Vector<int32_t, 3> v2 = {4, 5, 6};
        auto res = v1.product(v2);
        REQUIRE( res == 32 );
    }

    SECTION("Hadamard product") {
        nn::Vector<int32_t, 3> v2 = {4, 5, 6};
        auto v3 = v1.hadamardProduct(v2);
        REQUIRE( v3[0] == 4 );
        REQUIRE( v3[1] == 10 );
        REQUIRE( v3[2] == 18 );
    }
}

TEST_CASE("Matrixes") {

    nn::Matrix<int32_t, 2, 3> m1 = {{1, 2, 3}, {4, 5, 6}};

    SECTION("Initialization") {
        REQUIRE( m1[0][0] == 1 );
        REQUIRE( m1[0][1] == 2 );
        REQUIRE( m1[0][2] == 3 );
        REQUIRE( m1[1][0] == 4 );
        REQUIRE( m1[1][1] == 5 );
        REQUIRE( m1[1][2] == 6 );
    }

    SECTION("Partial initialization") {
        nn::Matrix<int32_t, 3, 3> m2 = {{1, 2}, {3}};
        REQUIRE( m2[0][0] == 1 );
        REQUIRE( m2[0][1] == 2 );
        REQUIRE( m2[0][2] == 0 );
        REQUIRE( m2[1][0] == 3 );
        REQUIRE( m2[1][1] == 0 );
        REQUIRE( m2[1][2] == 0 );
        REQUIRE( m2[2][0] == 0 );
        REQUIRE( m2[2][1] == 0 );
        REQUIRE( m2[2][2] == 0 );
    }

    SECTION("Fill") {
        nn::SequenceDistribution<int32_t> distribution(10, 2);
        m1.fill(distribution);
        REQUIRE( m1[0][0] == 10 );
        REQUIRE( m1[0][1] == 12 );
        REQUIRE( m1[0][2] == 14 );
        REQUIRE( m1[1][0] == 16 );
        REQUIRE( m1[1][1] == 18 );
        REQUIRE( m1[1][2] == 20 );
    }

    SECTION("Add") {
        nn::Matrix<int32_t, 2, 3> m2 = {{1, 2, 3}, {4, 5, 6}};
        nn::Matrix<int32_t, 2, 3> m3 = m1.add(m2);
        REQUIRE( m3[0][0] == 2 );
        REQUIRE( m3[0][1] == 4 );
        REQUIRE( m3[0][2] == 6 );
        REQUIRE( m3[1][0] == 8 );
        REQUIRE( m3[1][1] == 10 );
        REQUIRE( m3[1][2] == 12 );
    }

    SECTION("Product") {
        nn::Vector<int32_t, 3> v = {1, 2, 3};
        nn::Vector<int32_t, 2> res = m1.product(v);
        REQUIRE( res[0] == 14 );
        REQUIRE( res[1] == 32 );
    }

    SECTION("Hadamard product") {
        nn::Matrix<int32_t, 2, 3> m2 = {{1, 2, 3}, {4, 5, 6}};
        nn::Matrix<int32_t, 2, 3> m3 = m1.hadamardProduct(m2);
        REQUIRE( m3[0][0] == 1 );
        REQUIRE( m3[0][1] == 4 );
        REQUIRE( m3[0][2] == 9 );
        REQUIRE( m3[1][0] == 16 );
        REQUIRE( m3[1][1] == 25 );
        REQUIRE( m3[1][2] == 36 );
    }
}

TEST_CASE("DenseLayer") {

    nn::SequenceDistribution<int32_t> distribution(1, 1);
    nn::DenseLayer<int32_t, nn::Product<int32_t, 3>, 3, 2> layer;
    layer.fill(distribution);

    SECTION("Fill") {
        const nn::Matrix<int32_t, 2, 3> & weights = layer.weights();
        const nn::Vector<int32_t, 2> & biases = layer.biases();
        REQUIRE( weights[0][0] == 1 );
        REQUIRE( weights[0][1] == 2 );
        REQUIRE( weights[0][2] == 3 );
        REQUIRE( weights[1][0] == 4 );
        REQUIRE( weights[1][1] == 5 );
        REQUIRE( weights[1][2] == 6 );
        REQUIRE( biases[0] == 7 );
        REQUIRE( biases[1] == 8 );
    }

    SECTION("Forward pass") {
        nn::Vector<int32_t, 3> input = {1, 2, 3};
        nn::Vector<int32_t, 2> output = layer.forward(input);
        REQUIRE( output[0] == 63 );
        REQUIRE( output[1] == 120 );
    }
}
