#include <iostream>
#include <vector>
#include <random>
#include "nnSolver.h"
#include "DataManager.h"

int main() {
    std::cout << "Testing C++ to Python training integration..." << std::endl;
    
    // Create nnSolverManager instance
    nnSolverManager solver;
    

    DataManager data_manager;
    TrainingData training_data = data_manager.load_single_csv("train_data/training_poisson_data_RPM1000.csv");
    std::vector<nnInputData> inputs = training_data.inputs;
    std::vector<SC> outputs = training_data.outputs;
    std::cout << "输入数据行数: " << inputs.size() << std::endl;
    std::cout << "输出数据数量: " << outputs.size() << std::endl;

    try {
        // Test with custom configuration
        std::cout << "\nTesting with custom configuration..." << std::endl;
        ModelConfig config(5, 1, 16, 0.025f, "flexible_dnn", "standardization", "relu", true);
        solver.launch_python_train_with_config(
            inputs, outputs, config,
            "test_output", "cpp_trained_model",
            20, 10240  // epochs, batch_size
        );
        
        std::cout << "\n✅ All tests completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Press Enter to continue..." << std::endl;
    std::cin.get();

    
    return 0;
} 