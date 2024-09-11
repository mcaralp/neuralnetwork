
cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

##############################################################
# Find xxd
##############################################################

find_program(XXD xxd)

##############################################################
# Fetch the MNIST dataset and Catch2
##############################################################

include(FetchContent)

FetchContent_Declare(
    mnist
    GIT_REPOSITORY https://github.com/mrgloom/MNIST-dataset-in-different-formats.git
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)

FetchContent_MakeAvailable(mnist)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

##############################################################
# Define the MNIST library
##############################################################

set(MNIST_ROOT "${mnist_SOURCE_DIR}/data/Original dataset")
set(GENERATED "${CMAKE_BINARY_DIR}/generated")

set(TRAIN_IMAGES_DATA "${MNIST_ROOT}/train-images.idx3-ubyte")
set(TRAIN_LABELS_DATA "${MNIST_ROOT}/train-labels.idx1-ubyte")
set(TEST_IMAGES_DATA  "${MNIST_ROOT}/t10k-images.idx3-ubyte")
set(TEST_LABELS_DATA  "${MNIST_ROOT}/t10k-labels.idx1-ubyte")

set(TRAIN_IMAGES "${GENERATED}/TrainImagesMNIST.cxx")
set(TRAIN_LABELS "${GENERATED}/TrainLabelsMNIST.cxx")
set(TEST_IMAGES  "${GENERATED}/TestImagesMNIST.cxx")
set(TEST_LABELS  "${GENERATED}/TestLabelsMNIST.cxx")

file(MAKE_DIRECTORY ${GENERATED})

add_library(
    mnist
    ${TRAIN_IMAGES}
    ${TRAIN_LABELS}
    ${TEST_IMAGES}
    ${TEST_LABELS}
)

add_custom_command(
    OUTPUT ${TRAIN_IMAGES}
    COMMAND ${XXD} -n trainImagesMNIST -i ${TRAIN_IMAGES_DATA} > ${TRAIN_IMAGES}
    DEPENDS ${TRAIN_IMAGES_DATA}
)

add_custom_command(
    OUTPUT ${TRAIN_LABELS}
    COMMAND ${XXD} -n trainLabelsMNIST -i ${TRAIN_LABELS_DATA} > ${TRAIN_LABELS}
    DEPENDS ${TRAIN_LABELS_DATA}
)

add_custom_command(
    OUTPUT ${TEST_IMAGES}
    COMMAND ${XXD} -n testImagesMNIST -i ${TEST_IMAGES_DATA} > ${TEST_IMAGES}
    DEPENDS ${TEST_IMAGES_DATA}
)

add_custom_command(
    OUTPUT ${TEST_LABELS}
    COMMAND ${XXD} -n testLabelsMNIST -i ${TEST_LABELS_DATA} > ${TEST_LABELS}
    DEPENDS ${TEST_LABELS_DATA}
)

##############################################################
# Define the neural network library
##############################################################

add_library(
    neuralnetwork
    INTERFACE
)

target_include_directories(
    neuralnetwork
    INTERFACE
    include
)

##############################################################
# Define the executable
##############################################################

add_executable(
    ${CMAKE_PROJECT_NAME}
    src/main.cxx
)

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
    mnist
    neuralnetwork
)

##############################################################
# Define the tests
##############################################################

add_executable(
    ${CMAKE_PROJECT_NAME}_tests
    src/tests.cxx
)

target_link_libraries(
    ${CMAKE_PROJECT_NAME}_tests
    Catch2::Catch2WithMain
    mnist
    neuralnetwork
)
