#ifndef NNSOLVER_H
#define NNSOLVER_H

#include "DataManager.h"

#include <vector>
#include <string>
#include <optional>
#include <onnxruntime_cxx_api.h>

#define SC float

// Model configuration structure for C++
struct ModelConfig {
    int input_dim = 5;
    int output_dim = 1;
    int base_neurons = 16;
    float dropout_prob = 0.025f;
    std::string model_type = "flexible_dnn";
    std::string norm_type = "standardization";
    std::string activation = "relu";
    bool mix_norm = true;
    
    ModelConfig() = default;
    
    ModelConfig(int input_dim, int output_dim, int base_neurons = 16, 
                float dropout_prob = 0.025f, const std::string& model_type = "flexible_dnn",
                const std::string& norm_type = "standardization", 
                const std::string& activation = "relu", bool mix_norm = true)
        : input_dim(input_dim), output_dim(output_dim), base_neurons(base_neurons),
          dropout_prob(dropout_prob), model_type(model_type), norm_type(norm_type),
          activation(activation), mix_norm(mix_norm) {}
};

class nnSolverManager{
    public:
        Ort::Session* session = nullptr;

        nnSolverManager();
        ~nnSolverManager();
        
        // Python training interface
        void launch_python_train_with_config(std::vector<nnInputData> inputHost, 
                                           std::vector<SC> outputHost, 
                                           const ModelConfig& config,
                                           const std::string& save_dir = "models",
                                           const std::string& model_name = "trained_model",
                                           int epochs = 100,
                                           int batch_size = 32);
        
        // Model loading and prediction
        void load_model(const std::string& model_path);
        void predict_onnx(const int n, nnInputData* inputDevice, SC* outputDevice);
        
    private:
        // Helper functions for data conversion
        std::vector<std::vector<float>> convert_inputs_to_python(const std::vector<nnInputData>& inputs);
        std::vector<float> convert_outputs_to_python(const std::vector<SC>& outputs);
};

#endif // NNSOLVER_H