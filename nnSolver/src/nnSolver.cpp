#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <optional>
#include <cstdlib>  // for setenv

#include "nnSolver.h"

// Python embedding headers
#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define SC float

nnSolverManager::nnSolverManager(){
    session = nullptr;
}

nnSolverManager::~nnSolverManager(){
    if(session){
        delete session;
    }
}

void nnSolverManager::load_model(const std::string& model_path){
    if (!nnSolverManager::session) {
        static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nnSolverManager");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_options;
        // cuda_options.cuda_graph_enable = 1;  // 启用 CUDA Graph
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        nnSolverManager::session = new Ort::Session(env, model_path.c_str(), session_options);
    }
}

void nnSolverManager::predict_onnx(const int n, nnInputData* inputDevice,
    SC* outputDevice){
    if (inputDevice == nullptr || outputDevice == nullptr)
    {
        return;
    }


    // // 1. 初始化 ONNX Runtime
    // static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    // static Ort::SessionOptions session_options;
    // static Ort::Session* session = nullptr;
    // if (!session) {
    //     session_options.SetIntraOpNumThreads(1);
    //     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    //     OrtCUDAProviderOptions cuda_options;
    //     // cuda_options.cuda_graph_enable = 1;  // 启用 CUDA Graph
    //     session_options.AppendExecutionProvider_CUDA(cuda_options);
    //     session = new Ort::Session(env, "models/trained_model_20250729_161931/trained_model.onnx", session_options);
    // }

    if(nnSolverManager::session == nullptr){
        nnSolverManager::load_model("models/trained_model_20250729_161931/trained_model.onnx");
    }

    // 2. 构造 device tensor
    std::vector<int64_t> input_shape = {n, 5};
    std::vector<int64_t> output_shape = {n, 1};

    // 关键：用 CUDA 内存创建 MemoryInfo
    Ort::MemoryInfo memory_info_cuda("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

    // 3. 创建输入输出 Ort::Value，直接用 device pointer
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda, (float*)inputDevice, n * 5, input_shape.data(), input_shape.size()
    );
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        memory_info_cuda, (float*)outputDevice, n * 1, output_shape.data(), output_shape.size()
    );

    // 4. 推理
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    std::array<Ort::Value, 1> input_tensors = {std::move(input_tensor)};
    std::array<Ort::Value, 1> output_tensors = {std::move(output_tensor)};

    nnSolverManager::session->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 1,
        output_names, output_tensors.data(), 1
    );
}

// Helper functions for data conversion
std::vector<std::vector<float>> nnSolverManager::convert_inputs_to_python(const std::vector<nnInputData>& inputs) {
    std::vector<std::vector<float>> python_inputs;
    python_inputs.reserve(inputs.size());
    
    for (const auto& input : inputs) {
        std::vector<float> row = {
            input.numberDensity_,
            input.divU_,
            input.APi_,
            input.APjSum_,
            input.ghostWeightP_
        };
        python_inputs.push_back(row);
    }
    
    return python_inputs;
}

std::vector<float> nnSolverManager::convert_outputs_to_python(const std::vector<SC>& outputs) {
    return std::vector<float>(outputs.begin(), outputs.end());
}


void nnSolverManager::launch_python_train_with_config(std::vector<nnInputData> inputHost, 
                                                     std::vector<SC> outputHost, 
                                                     const ModelConfig& config,
                                                     const std::string& save_dir,
                                                     const std::string& model_name,
                                                     int epochs,
                                                     int batch_size) {
    try {

        // 使用CMake传递的Python路径设置环境变量
        #ifdef PYTHON_ROOT_DIR
        std::string python_root = PYTHON_ROOT_DIR;
        std::string python_version = PYTHON_VERSION;
        
        // 设置PYTHONHOME
        setenv("PYTHONHOME", python_root.c_str(), 1);
        
        // 设置PYTHONPATH
        std::string python_path = python_root + "/lib/python" + python_version + ":" + 
                                 python_root + "/lib/python" + python_version + "/site-packages";
        setenv("PYTHONPATH", python_path.c_str(), 1);
        
        // 设置LD_LIBRARY_PATH
        std::string ld_library_path = python_root + "/lib:" + 
                                     (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "");
        setenv("LD_LIBRARY_PATH", ld_library_path.c_str(), 1);
        
        std::cout << "Set PYTHONHOME to: " << python_root << std::endl;
        std::cout << "Set PYTHONPATH to: " << python_path << std::endl;
        std::cout << "Set LD_LIBRARY_PATH to: " << ld_library_path << std::endl;
        #endif
        
        // Initialize Python interpreter
        py::scoped_interpreter guard{};
        
        // Add py_solver to Python path - use absolute path
        py::module sys = py::module::import("sys");
        py::module os = py::module::import("os");
        
        // Get current working directory and construct absolute path to py_solver
        py::str current_dir = os.attr("getcwd")();
        std::string current_dir_str = py::str(current_dir).cast<std::string>();
        std::string py_solver_path_str = current_dir_str + "/py_solver";
        
        // Add to Python path
        py::list path_list = sys.attr("path");
        path_list.append(py_solver_path_str);
        
        // Also add conda environment path if available
        // if (conda_prefix) {
        //     std::string conda_path = std::string(conda_prefix) + "/lib/python3.10/site-packages";
        //     path_list.append(conda_path);
        //     std::cout << "Added conda path: " << conda_path << std::endl;
            
        //     // Also add the conda environment's Python executable directory
        //     std::string conda_python_path = std::string(conda_prefix) + "/lib/python3.10";
        //     path_list.append(conda_python_path);
        //     std::cout << "Added conda python path: " << conda_python_path << std::endl;
        // }
        
        // std::cout << "Added to Python path: " << py_solver_path_str << std::endl;
        
        // Import Python modules with error checking
        py::module pipeline_module;
        py::module config_module;
        py::module py_solver_module;
        
        try {
            // Import the py_solver package first
            py_solver_module = py::module::import("py_solver");
            
            // Then import modules from the package
            pipeline_module = py::module::import("py_solver.pipeline_manager");
            config_module = py::module::import("py_solver.config");
            std::cout << "Successfully imported Python modules" << std::endl;
        } catch (const py::error_already_set& e) {
            std::cerr << "Failed to import Python modules: " << e.what() << std::endl;
            throw;
        }
        
        // Convert C++ data to Python format
        std::vector<std::vector<float>> python_inputs = convert_inputs_to_python(inputHost);
        std::vector<float> python_outputs = convert_outputs_to_python(outputHost);
        
        // Convert to numpy arrays using proper method
        std::cout << "Converting data to numpy arrays..." << std::endl;
        
        // Create numpy arrays using proper constructor
        std::vector<py::ssize_t> input_shape = {static_cast<py::ssize_t>(python_inputs.size()), 
                                               static_cast<py::ssize_t>(python_inputs[0].size())};
        std::vector<py::ssize_t> output_shape = {static_cast<py::ssize_t>(python_outputs.size()), 1};
        
        py::array_t<float> inputs_array(input_shape);
        py::array_t<float> outputs_array(output_shape);
        
        // Get buffer info
        auto inputs_buf = inputs_array.request();
        auto outputs_buf = outputs_array.request();
        
        // Get pointers to data
        float* inputs_ptr = static_cast<float*>(inputs_buf.ptr);
        float* outputs_ptr = static_cast<float*>(outputs_buf.ptr);
        
        // Copy data to numpy arrays
        for (size_t i = 0; i < python_inputs.size(); ++i) {
            for (size_t j = 0; j < python_inputs[i].size(); ++j) {
                inputs_ptr[i * python_inputs[i].size() + j] = python_inputs[i][j];
            }
        }
        
        for (size_t i = 0; i < python_outputs.size(); ++i) {
            outputs_ptr[i] = python_outputs[i];
        }
        
        std::cout << "Data conversion completed. Input shape: " << python_inputs.size() 
                  << "x" << python_inputs[0].size() << ", Output shape: " << python_outputs.size() << "x1" << std::endl;
        
        // Create Python model config
        py::object model_config_class = config_module.attr("model_config");
        py::object py_config = model_config_class(
            config.input_dim,
            config.output_dim,
            config.base_neurons,
            config.dropout_prob,
            config.model_type,
            config.norm_type,
            py::none(),  // norm_info
            config.activation,
            config.mix_norm
        );
        
        // Create pipeline manager
        py::object pipeline_class = pipeline_module.attr("PipelineManager");
        py::object pipeline = pipeline_class(py_config);
        
        // Call training function
        py::object results = pipeline.attr("train_and_save_online")(
            inputs_array,
            outputs_array,
            epochs,
            batch_size,
            0.2,  // test_size
            save_dir,
            model_name
        );
        
        std::cout << "Python training completed successfully!" << std::endl;
        
        // Get save directory before Python objects are destroyed
        std::string save_dir_result = py::str(results["save_dir"]).cast<std::string>();
        std::cout << "Model saved to: " << save_dir_result << std::endl;
        
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error during training: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "C++ error during training: " << e.what() << std::endl;
        throw;
    }
}

