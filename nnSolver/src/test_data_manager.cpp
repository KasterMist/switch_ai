#include "DataManager.h"
#include "nnSolver.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>


int main() {
    DataManager data_manager;
    
    std::cout << "=== 测试1：加载包含target列的CSV文件 ===" << std::endl;
    auto result = data_manager.load_single_csv("test_data/training_poisson_data_RPM1000.csv");
    
    std::cout << "输入数据行数: " << result.inputs.size() << std::endl;
    std::cout << "输出数据数量: " << result.outputs.size() << std::endl;

    std::vector<nnInputData> inputs = result.inputs;
    std::vector<SC> outputs = std::vector<SC>(result.num_samples);
    nnInputData* inputDevice;
    SC* outputDevice;
    cudaMalloc(&inputDevice, inputs.size() * sizeof(nnInputData));
    cudaMalloc(&outputDevice, outputs.size() * sizeof(SC));
    cudaMemcpy(inputDevice, inputs.data(), inputs.size() * sizeof(nnInputData), cudaMemcpyHostToDevice);
    cudaMemcpy(outputDevice, outputs.data(), outputs.size() * sizeof(SC), cudaMemcpyHostToDevice);
    nnSolverManager manager;
    manager.predict_onnx(inputs.size(), inputDevice, outputDevice);
    // predict_onnx(inputs.size(), inputDevice, outputDevice);
    cudaMemcpy(outputs.data(), outputDevice, outputs.size() * sizeof(SC), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; i++){
        std::cout << "预测值: " << outputs[i] << " 真实值: " << result.outputs[i] << std::endl;
    }
    std::cout << "success" << std::endl;
    cudaFree(inputDevice);
    cudaFree(outputDevice);
} 