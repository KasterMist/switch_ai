cmake_minimum_required(VERSION 3.22.0)

project(nnSolver LANGUAGES C CXX)

if(EXISTS "${CMAKE_SOURCE_DIR}/load_config.cmake")
    include("${CMAKE_SOURCE_DIR}/load_config.cmake")
else()
    message(FATAL_ERROR "load_config.cmake not found! Please write your load_config.cmake.")
endif()


add_executable(nnSolver nnSolver/src/nnSolver.cpp nnSolver/src/DataManager.cpp nnSolver/src/test_python_training.cpp)

# Torch
find_package(Torch REQUIRED)
target_link_libraries(nnSolver PRIVATE ${TORCH_LIBRARIES})
target_include_directories(nnSolver PRIVATE ${TORCH_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/nnSolver/include)

# pybind11
find_package(pybind11 REQUIRED)
target_link_libraries(nnSolver PRIVATE pybind11::embed)

# ONNX
find_package(ONNX REQUIRED)
target_link_libraries(nnSolver PRIVATE ${ONNX_LIBRARIES})
target_include_directories(nnSolver PRIVATE ${ONNX_INCLUDE_DIRS})

# onnxruntime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h)
find_library(ONNXRUNTIME_LIB onnxruntime)
target_include_directories(nnSolver PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(nnSolver PRIVATE ${ONNXRUNTIME_LIB})

# Python - Use conda environment
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
target_include_directories(nnSolver PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(nnSolver PRIVATE ${Python3_LIBRARIES})

# Set Python executable path to conda environment
set(PYTHON_EXECUTABLE ${Python3_ROOT_DIR}/bin/python)
set(PYTHON_LIBRARY ${Python3_ROOT_DIR}/lib/libpython${Python3_VERSION}.so)
set(PYTHON_INCLUDE_DIR ${Python3_ROOT_DIR}/include/python${Python3_VERSION})

# Add compile definitions to ensure correct Python environment
target_compile_definitions(nnSolver PRIVATE 
    PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"
    PYTHON_LIBRARY="${PYTHON_LIBRARY}"
    PYTHON_INCLUDE_DIR="${PYTHON_INCLUDE_DIR}"
    PYTHON_ROOT_DIR="${Python3_ROOT_DIR}"
    PYTHON_VERSION="${Python3_VERSION}"
)

# CUDA
find_package(CUDA REQUIRED)
target_link_libraries(nnSolver PRIVATE CUDA::cudart)
target_include_directories(nnSolver PRIVATE ${CUDA_INCLUDE_DIRS})

