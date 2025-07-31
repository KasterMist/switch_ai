#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <numeric>

#include <torch/torch.h>
#include <torch/script.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <onnxruntime_cxx_api.h>

#include "nnSolver.h"

#define SC float

namespace py = pybind11;

PYBIND11_MODULE(nnSolver, m){
    py::class_<nnInputData>(m, "nnInputData")
        .def(py::init<>())
        .def_readwrite("numberDensity_", &nnInputData::numberDensity_)
        .def_readwrite("divU_", &nnInputData::divU_)
        .def_readwrite("APi_", &nnInputData::APi_)
        .def_readwrite("APjSum_", &nnInputData::APjSum_)
        .def_readwrite("ghostWeightP_", &nnInputData::ghostWeightP_);
}

ExpDecayActivation::ExpDecayActivation()
{
    e = 2.718281828459045;  // Euler's number
}

torch::Tensor ExpDecayActivation::forward(torch::Tensor x)
{
    return x * torch::exp(-x.pow(2) / (2 * e));
}


DNNWithNorm::DNNWithNorm(const int input_dim, 
            const int base_neurons, 
            const SC dropout_prob,
            torch::Tensor input_mean,
            torch::Tensor input_std,
            torch::Tensor target_mean,
            torch::Tensor target_std)
{
    // Register linear layers
    layer1 = register_module("layer1",
                                torch::nn::Linear(input_dim, 8 * base_neurons));
    layer2 = register_module(
        "layer2", torch::nn::Linear(8 * base_neurons, 4 * base_neurons));
    layer3 = register_module(
        "layer3", torch::nn::Linear(4 * base_neurons, 2 * base_neurons));
    output_layer = register_module("output_layer",
                                    torch::nn::Linear(2 * base_neurons, 1));

    // Register activation function and dropout
    activation = ExpDecayActivation();
    dropout = register_module("dropout", torch::nn::Dropout(dropout_prob));

    // Initialize weights
    init_weights();

    // Register normalization parameters as buffers
    torch::Tensor input_mean_tensor = input_mean;
    torch::Tensor input_std_tensor = input_std;
    torch::Tensor target_mean_tensor = target_mean;
    torch::Tensor target_std_tensor = target_std;
    
    if (input_mean_tensor.numel() == 0) {
        input_mean_tensor = torch::zeros({input_dim}, torch::kFloat32);
    }
    if (input_std_tensor.numel() == 0) {
        input_std_tensor = torch::ones({input_dim}, torch::kFloat32);
    }
    if (target_mean_tensor.numel() == 0) {
        target_mean_tensor = torch::zeros({1}, torch::kFloat32);
    }
    if (target_std_tensor.numel() == 0) {
        target_std_tensor = torch::ones({1}, torch::kFloat32);
    }

    register_buffer("input_mean", input_mean_tensor);
    register_buffer("input_std", input_std_tensor);
    register_buffer("target_mean", target_mean_tensor);
    register_buffer("target_std", target_std_tensor);
}

torch::Tensor DNNWithNorm::forward(torch::Tensor x)
{
    // Get normalization parameters
    torch::Tensor input_mean = this->named_buffers()["input_mean"];
    torch::Tensor input_std = this->named_buffers()["input_std"];
    torch::Tensor target_mean = this->named_buffers()["target_mean"];
    torch::Tensor target_std = this->named_buffers()["target_std"];
    
    // Automatically normalize input
    x = (x - input_mean) / input_std;
    
    x = activation.forward(layer1(x));
    x = dropout(x);
    x = activation.forward(layer2(x));
    x = dropout(x);
    x = activation.forward(layer3(x));
    x = dropout(x);
    x = output_layer(x);
    
    // Automatically denormalize output
    x = x * target_std + target_mean;
    return x;
}

void DNNWithNorm::init_weights()
{
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            torch::nn::init::xavier_uniform_(linear->weight);
            torch::nn::init::zeros_(linear->bias);
        }
    }
}

void DNNWithNorm::reset_parameters()
{
    layer1->reset_parameters();
    layer2->reset_parameters();
    layer3->reset_parameters();
    output_layer->reset_parameters();
    init_weights();
}

void DNNWithNorm::set_normalization_params(torch::Tensor input_mean, torch::Tensor input_std,
                                 torch::Tensor target_mean, torch::Tensor target_std)
{
    register_buffer("input_mean", input_mean);
    register_buffer("input_std", input_std);
    register_buffer("target_mean", target_mean);
    register_buffer("target_std", target_std);
}



PressureSolverTrainer::PressureSolverTrainer(double learning_rate,
                        double weight_decay,
                        int batch_size,
                        int num_epochs,
                        double dropout_prob,
                        int base_neurons)
    : learning_rate_(learning_rate),
        weight_decay_(weight_decay),
        batch_size_(batch_size),
        num_epochs_(num_epochs),
        dropout_prob_(dropout_prob),
        base_neurons_(base_neurons),
        device_(torch::kCUDA)  // Default to GPU
{
    std::cout << "Configuration: device=GPU, batch_size=" << batch_size_ 
                << ", base_neurons=" << base_neurons_ << std::endl;
}

// Load single CSV file
// Helper function: clean string and convert to float (supports scientific notation)
SC PressureSolverTrainer::safe_stof(const std::string& str, const std::string& field_name, int line_num) {
    std::string cleaned = str;
    
    // Remove leading and trailing whitespace
    cleaned.erase(0, cleaned.find_first_not_of(" \t\r\n"));
    cleaned.erase(cleaned.find_last_not_of(" \t\r\n") + 1);
    
    // Remove quotes
    if (!cleaned.empty() && (cleaned.front() == '"' || cleaned.front() == '\'')) {
        cleaned.erase(0, 1);
    }
    if (!cleaned.empty() && (cleaned.back() == '"' || cleaned.back() == '\'')) {
        cleaned.erase(cleaned.size() - 1, 1);
    }
    
    // Check if empty
    if (cleaned.empty()) {
        throw std::runtime_error("Line " + std::to_string(line_num) + ", " + field_name + " field is empty");
    }
    
    // Check if scientific notation format
    bool is_scientific = false;
    size_t e_pos = cleaned.find('e');
    if (e_pos == std::string::npos) {
        e_pos = cleaned.find('E');  // Also support uppercase E
    }
    if (e_pos != std::string::npos) {
        is_scientific = true;
    }
    
    try {
        // Use std::stod to support scientific notation, then convert to float
        double value = std::stod(cleaned);
        
        // Check if extremely small value (possible numerical precision issue)
        if (std::abs(value) < 1e-40) {
            // For extremely small values, set to 0 directly
            return 0.0f;
        }
        
        return static_cast<SC>(value);
    } catch (const std::exception& e) {
        throw std::runtime_error("Line " + std::to_string(line_num) + ", " + field_name + 
                                " field conversion failed: '" + cleaned + "', error: " + e.what());
    }
}

TrainingData PressureSolverTrainer::load_single_csv(const std::string& csv_file) {
    std::cout << "Loading: " << csv_file << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + csv_file);
    }

    std::vector<nnInputData> input_data;
    std::vector<SC> target_data;
    std::string line;
    int line_num = 0;
    int error_count = 0;
    const int max_errors = 10; // Maximum display 10 errors
    
    // Skip header line
    std::getline(file, line);
    line_num++;
    
    while (std::getline(file, line)) {
        line_num++;
        
        try {
            std::stringstream ss(line);
            std::string cell;
            nnInputData input;
            
            // Read first 5 columns as features
            std::getline(ss, cell, ',');
            input.numberDensity_ = safe_stof(cell, "numberDensity", line_num);
            
            std::getline(ss, cell, ',');
            input.divU_ = safe_stof(cell, "divU", line_num);
            
            std::getline(ss, cell, ',');
            input.APi_ = safe_stof(cell, "APi", line_num);
            
            std::getline(ss, cell, ',');
            input.APjSum_ = safe_stof(cell, "APjSum", line_num);
            
            std::getline(ss, cell, ',');
            input.ghostWeightP_ = safe_stof(cell, "ghostWeightP", line_num);
            
            // Read 6th column as target
            std::getline(ss, cell, ',');
            SC target = safe_stof(cell, "pressure", line_num);
            
            // Filter pressure range
            if (target >= -3000 && target <= 5000) {
                input_data.push_back(input);
                target_data.push_back(target);
            }
            
        } catch (const std::exception& e) {
            error_count++;
            if (error_count <= max_errors) {
                std::cout << "Line " << line_num << " parsing error: " << e.what() << std::endl;
            } else if (error_count == max_errors + 1) {
                std::cout << "Too many errors, stopping detailed error display..." << std::endl;
            }
            continue; // Skip this line, continue processing next line
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Loading time: " << duration.count() / 1000.0 
                << "s, data size: " << input_data.size() << " samples";
    if (error_count > 0) {
        std::cout << ", skipped " << error_count << " error rows";
    }
    std::cout << std::endl;
    
    return {input_data, target_data, static_cast<int>(input_data.size())};
}

// Load CSV data (supports multiple files)
TrainingData PressureSolverTrainer::load_csv_data(const std::vector<std::string>& csv_files) {
    std::cout << "Starting to load " << csv_files.size() << " data files..." << std::endl;
    auto total_start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<nnInputData> all_input_data;
    std::vector<SC> all_target_data;
    int total_samples = 0;
    
    for (const auto& csv_file : csv_files) {
        if (!std::filesystem::exists(csv_file)) {
            throw std::runtime_error("File does not exist: " + csv_file);
        }
        
        TrainingData file_data = load_single_csv(csv_file);
        all_input_data.insert(all_input_data.end(), file_data.input_data.begin(), file_data.input_data.end());
        all_target_data.insert(all_target_data.end(), file_data.target_data.begin(), file_data.target_data.end());
        total_samples += file_data.num_samples;
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    
    std::cout << "Data loading completed: " << total_samples << " samples, total time: " 
                << total_duration.count() / 1000.0 << "s" << std::endl;
    std::cout << "Pressure range: " << *std::min_element(all_target_data.begin(), all_target_data.end()) 
                << " ~ " << *std::max_element(all_target_data.begin(), all_target_data.end()) << std::endl;
    
    return {all_input_data, all_target_data, total_samples};
}

// Variadic template: supports any number of string arguments
template<typename... Args>
TrainingData PressureSolverTrainer::load_csv_data(const std::string& first_file, Args... args) {
    std::vector<std::string> files = {first_file, args...};
    return load_csv_data(files);
}

// Overload: supports single file string
TrainingData PressureSolverTrainer::load_csv_data(const std::string& csv_file) {
    return load_csv_data(std::vector<std::string>{csv_file});
}

// Variadic template: supports any number of string arguments for training
template<typename... Args>
void PressureSolverTrainer::train(const std::string& first_file, Args... args) {
    std::vector<std::string> files = {first_file, args...};
    train(files);
}



// Overload: supports single file string training
void PressureSolverTrainer::train(const std::string& csv_file) {
    train(std::vector<std::string>{csv_file});
}

// Compute normalization parameters
void PressureSolverTrainer::compute_normalization_params(const std::vector<nnInputData>& input_data, 
                                const std::vector<SC>& target_data) {
    std::cout << "Computing normalization parameters..." << std::endl;
    
    int num_samples = input_data.size();
    
    // Initialize means
    norm_params_.input_mean = {0, 0, 0, 0, 0};
    norm_params_.target_mean = 0.0;
    
    // Calculate means
    for (int i = 0; i < num_samples; ++i) {
        const auto& item = input_data[i];
        norm_params_.input_mean.numberDensity_ += item.numberDensity_ / num_samples;
        norm_params_.input_mean.divU_ += item.divU_ / num_samples;
        norm_params_.input_mean.APi_ += item.APi_ / num_samples;
        norm_params_.input_mean.APjSum_ += item.APjSum_ / num_samples;
        norm_params_.input_mean.ghostWeightP_ += item.ghostWeightP_ / num_samples;
        norm_params_.target_mean += target_data[i] / num_samples;
    }
    
    // Initialize standard deviations
    norm_params_.input_std = {0, 0, 0, 0, 0};
    norm_params_.target_std = 0.0;
    
    // Calculate standard deviations
    for (int i = 0; i < num_samples; ++i) {
        const auto& item = input_data[i];
        SC diff;
        
        diff = item.numberDensity_ - norm_params_.input_mean.numberDensity_;
        norm_params_.input_std.numberDensity_ += diff * diff;
        diff = item.divU_ - norm_params_.input_mean.divU_;
        norm_params_.input_std.divU_ += diff * diff;
        diff = item.APi_ - norm_params_.input_mean.APi_;
        norm_params_.input_std.APi_ += diff * diff;
        diff = item.APjSum_ - norm_params_.input_mean.APjSum_;
        norm_params_.input_std.APjSum_ += diff * diff;
        diff = item.ghostWeightP_ - norm_params_.input_mean.ghostWeightP_;
        norm_params_.input_std.ghostWeightP_ += diff * diff;
        
        diff = target_data[i] - norm_params_.target_mean;
        norm_params_.target_std += diff * diff;
    }
    
    // Calculate standard deviations
    norm_params_.input_std.numberDensity_ = std::sqrt(norm_params_.input_std.numberDensity_ / num_samples);
    norm_params_.input_std.divU_ = std::sqrt(norm_params_.input_std.divU_ / num_samples);
    norm_params_.input_std.APi_ = std::sqrt(norm_params_.input_std.APi_ / num_samples);
    norm_params_.input_std.APjSum_ = std::sqrt(norm_params_.input_std.APjSum_ / num_samples);
    norm_params_.input_std.ghostWeightP_ = std::sqrt(norm_params_.input_std.ghostWeightP_ / num_samples);
    norm_params_.target_std = std::sqrt(norm_params_.target_std / num_samples);
    
    // Save normalization parameters to file
    save_normalization_params();
    
    std::cout << "Normalization parameters computation completed" << std::endl;
}

// Save normalization parameters
void PressureSolverTrainer::save_normalization_params() {
    std::ofstream os("data-parameters.txt");
    if (!os.is_open()) {
        throw std::runtime_error("Failed to create normalization parameters file");
    }
    
    os << std::scientific << std::setprecision(7) << std::showpos;
    os << norm_params_.input_mean.numberDensity_ << " " << norm_params_.input_std.numberDensity_ << std::endl;
    os << norm_params_.input_mean.divU_ << " " << norm_params_.input_std.divU_ << std::endl;
    os << norm_params_.input_mean.APi_ << " " << norm_params_.input_std.APi_ << std::endl;
    os << norm_params_.input_mean.APjSum_ << " " << norm_params_.input_std.APjSum_ << std::endl;
    os << norm_params_.input_mean.ghostWeightP_ << " " << norm_params_.input_std.ghostWeightP_ << std::endl;
    os << norm_params_.target_mean << " " << norm_params_.target_std << std::endl;
    os.close();
    
    std::cout << "Normalization parameters saved to data-parameters.txt" << std::endl;
}

// Load normalization parameters
void PressureSolverTrainer::load_normalization_params() {
    std::ifstream is("data-parameters.txt");
    if (!is.is_open()) {
        throw std::runtime_error("Normalization parameters file not found: data-parameters.txt");
    }
    
    is >> norm_params_.input_mean.numberDensity_ >> norm_params_.input_std.numberDensity_;
    is >> norm_params_.input_mean.divU_ >> norm_params_.input_std.divU_;
    is >> norm_params_.input_mean.APi_ >> norm_params_.input_std.APi_;
    is >> norm_params_.input_mean.APjSum_ >> norm_params_.input_std.APjSum_;
    is >> norm_params_.input_mean.ghostWeightP_ >> norm_params_.input_std.ghostWeightP_;
    is >> norm_params_.target_mean >> norm_params_.target_std;
    is.close();
    
    std::cout << "Normalization parameters loaded" << std::endl;
}

// data loader
std::vector<Batch> PressureSolverTrainer::data_loader(const torch::Tensor& input_tensor, 
                                const torch::Tensor& target_tensor,
                                const int batch_size) {
    const int num_samples = input_tensor.size(0);
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    // shuffle data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    torch::Tensor random_indices = torch::from_blob(indices.data(), {num_samples}, torch::kInt).to(torch::kLong);
    
    std::vector<Batch> data_set;
    for (int i = 0; i < num_samples; i += batch_size) {
        Batch batch;
        torch::Tensor training_indices = random_indices.index(
            {torch::indexing::Slice(i, std::min(num_samples, i + batch_size))});
        batch.data_ = input_tensor.index({training_indices}).to(device_);
        batch.target_ = target_tensor.index({training_indices}).to(device_);
        data_set.push_back(batch);
    }
    return data_set;
}

// convert input data to tensor
torch::Tensor PressureSolverTrainer::input_data_to_tensor(const std::vector<nnInputData>& input_data) {
    int num_samples = input_data.size();
    torch::Tensor tensor = torch::zeros({num_samples, 5}, torch::kFloat32);
    
    for (int i = 0; i < num_samples; ++i) {
        const auto& item = input_data[i];
        tensor[i][0] = item.numberDensity_;
        tensor[i][1] = item.divU_;
        tensor[i][2] = item.APi_;
        tensor[i][3] = item.APjSum_;
        tensor[i][4] = item.ghostWeightP_;
    }
    
    return tensor;
}

// convert target ata to tensor
torch::Tensor PressureSolverTrainer::target_data_to_tensor(const std::vector<SC>& target_data) {
    int num_samples = target_data.size();
    torch::Tensor tensor = torch::zeros({num_samples, 1}, torch::kFloat32);
    
    for (int i = 0; i < num_samples; ++i) {
        tensor[i][0] = target_data[i];
    }
    
    return tensor;
}

void PressureSolverTrainer::train_data(TrainingData& data){
    float test_size_ratio = 0.2;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // split train/val set
    std::cout << "Split train/val set..." << std::endl;
    int num_train = static_cast<int>(data.num_samples * (1 - test_size_ratio));
    
    std::vector<nnInputData> train_input(data.input_data.begin(), data.input_data.begin() + num_train);
    std::vector<SC> train_target(data.target_data.begin(), data.target_data.begin() + num_train);
    std::vector<nnInputData> val_input(data.input_data.begin() + num_train, data.input_data.end());
    std::vector<SC> val_target(data.target_data.begin() + num_train, data.target_data.end());
    
    // compute normalization parameters
    compute_normalization_params(train_input, train_target);
    
    // create model
    std::cout << "Create neural network model..." << std::endl;
    
    // convert normalization parameters to tensor
    torch::Tensor input_mean_tensor = torch::zeros({5}, torch::kFloat32);
    torch::Tensor input_std_tensor = torch::zeros({5}, torch::kFloat32);
    
    input_mean_tensor[0] = norm_params_.input_mean.numberDensity_;
    input_mean_tensor[1] = norm_params_.input_mean.divU_;
    input_mean_tensor[2] = norm_params_.input_mean.APi_;
    input_mean_tensor[3] = norm_params_.input_mean.APjSum_;
    input_mean_tensor[4] = norm_params_.input_mean.ghostWeightP_;
    
    input_std_tensor[0] = norm_params_.input_std.numberDensity_;
    input_std_tensor[1] = norm_params_.input_std.divU_;
    input_std_tensor[2] = norm_params_.input_std.APi_;
    input_std_tensor[3] = norm_params_.input_std.APjSum_;
    input_std_tensor[4] = norm_params_.input_std.ghostWeightP_;
    
    torch::Tensor target_mean_tensor = torch::tensor({norm_params_.target_mean}, torch::kFloat32);
    torch::Tensor target_std_tensor = torch::tensor({norm_params_.target_std}, torch::kFloat32);
    
    model_ = std::make_shared<DNNWithNorm>(
        5, base_neurons_, dropout_prob_, 
        input_mean_tensor, input_std_tensor, target_mean_tensor, target_std_tensor
    );
    model_->to(device_);
    
    // optimizer and loss function
    torch::optim::AdamW optimizer(model_->parameters(), 
                                    torch::optim::AdamWOptions(learning_rate_)
                                        .weight_decay(weight_decay_)
                                        .amsgrad(true)
                                        .eps(1e-5));
    
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/5, /*gamma=*/0.9);
    
    torch::nn::MSELoss criterion;
    
    // convert to tensor
    torch::Tensor train_input_tensor = input_data_to_tensor(train_input).to(device_);
    torch::Tensor train_target_tensor = target_data_to_tensor(train_target).to(device_);
    torch::Tensor val_input_tensor = input_data_to_tensor(val_input).to(device_);
    torch::Tensor val_target_tensor = target_data_to_tensor(val_target).to(device_);
    
    // train loop
    std::cout << "️Start training loop..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (int epoch = 0; epoch < num_epochs_; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // train phase
        model_->train();
        double train_loss = 0.0;
        int train_batches = 0;
        
        auto batches = data_loader(train_input_tensor, train_target_tensor, batch_size_);
        const int num_batches = batches.size();
        const int train_batch_count = static_cast<int>(num_batches * 0.8);

        // get normalization parameters
        torch::Tensor target_mean_tensor = model_->named_buffers()["target_mean"];
        torch::Tensor target_std_tensor = model_->named_buffers()["target_std"];
        
        for (int i = 0; i < train_batch_count; ++i) {
            optimizer.zero_grad();
            torch::Tensor predictions = model_->forward(batches[i].data_);
            // normalization loss
            torch::Tensor predictions_norm = (predictions - target_mean_tensor) / target_std_tensor;
            torch::Tensor targets_norm = (batches[i].target_ - target_mean_tensor) / target_std_tensor;
            torch::Tensor loss = criterion(predictions_norm, targets_norm);
            
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
            optimizer.step();
            
            train_loss += loss.item<float>();
            train_batches++;
        }
        
        scheduler.step();
        
        // validation phase
        model_->eval();
        double val_loss = 0.0;
        double val_mae = 0.0;
        int val_batches = 0;
        
        torch::NoGradGuard no_grad;
        for (int i = train_batch_count; i < num_batches; ++i) {
            torch::Tensor predictions = model_->forward(batches[i].data_);
            // normalization loss
            torch::Tensor predictions_norm = (predictions - target_mean_tensor) / target_std_tensor;
            torch::Tensor targets_norm = (batches[i].target_ - target_mean_tensor) / target_std_tensor;
            torch::Tensor loss = criterion(predictions_norm, targets_norm);
            
            val_loss += loss.item<float>();
            val_mae += torch::mean(torch::abs(predictions - batches[i].target_)).item<float>();
            val_batches++;
        }
        
        double avg_train_loss = train_loss / train_batches;
        double avg_val_loss = val_loss / val_batches;
        double avg_val_mae = val_mae / val_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        // if ((epoch + 1) % 5 == 0 || epoch == 0) {
        std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << num_epochs_ << ": "
                    << "Loss=" << std::fixed << std::setprecision(6) << avg_train_loss << ", "
                    << "Val_Loss=" << avg_val_loss << ", "
                    << "MAE=" << avg_val_mae << ", "
                    << "Time=" << epoch_duration.count() / 1000.0 << "s" << std::endl;
        // }
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Training completed!" << std::endl;
    std::cout << "Total time: " << total_duration.count() / 1000.0 << "s" << std::endl;
}

// train model (supports multiple CSV files)
void PressureSolverTrainer::train(const std::vector<std::string>& csv_files) {
    float test_size_ratio = 0.2;
    auto total_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Start training..." << std::endl;
    
    // load data
    TrainingData data = load_csv_data(csv_files);

    train_data(data);
    return;
    // split train/val set
    std::cout << "Split train/val set..." << std::endl;
    int num_train = static_cast<int>(data.num_samples * (1 - test_size_ratio));
    
    std::vector<nnInputData> train_input(data.input_data.begin(), data.input_data.begin() + num_train);
    std::vector<SC> train_target(data.target_data.begin(), data.target_data.begin() + num_train);
    std::vector<nnInputData> val_input(data.input_data.begin() + num_train, data.input_data.end());
    std::vector<SC> val_target(data.target_data.begin() + num_train, data.target_data.end());
    
    // compute normalization parameters
    compute_normalization_params(train_input, train_target);
    
    // create model
    std::cout << "Create neural network model..." << std::endl;
    
    // convert normalization parameters to tensor
    torch::Tensor input_mean_tensor = torch::zeros({5}, torch::kFloat32);
    torch::Tensor input_std_tensor = torch::zeros({5}, torch::kFloat32);
    
    input_mean_tensor[0] = norm_params_.input_mean.numberDensity_;
    input_mean_tensor[1] = norm_params_.input_mean.divU_;
    input_mean_tensor[2] = norm_params_.input_mean.APi_;
    input_mean_tensor[3] = norm_params_.input_mean.APjSum_;
    input_mean_tensor[4] = norm_params_.input_mean.ghostWeightP_;
    
    input_std_tensor[0] = norm_params_.input_std.numberDensity_;
    input_std_tensor[1] = norm_params_.input_std.divU_;
    input_std_tensor[2] = norm_params_.input_std.APi_;
    input_std_tensor[3] = norm_params_.input_std.APjSum_;
    input_std_tensor[4] = norm_params_.input_std.ghostWeightP_;
    
    torch::Tensor target_mean_tensor = torch::tensor({norm_params_.target_mean}, torch::kFloat32);
    torch::Tensor target_std_tensor = torch::tensor({norm_params_.target_std}, torch::kFloat32);
    
    model_ = std::make_shared<DNNWithNorm>(
        5, base_neurons_, dropout_prob_, 
        input_mean_tensor, input_std_tensor, target_mean_tensor, target_std_tensor
    );
    model_->to(device_);
    
    // optimizer and loss function
    torch::optim::AdamW optimizer(model_->parameters(), 
                                    torch::optim::AdamWOptions(learning_rate_)
                                        .weight_decay(weight_decay_)
                                        .amsgrad(true)
                                        .eps(1e-5));
    
    torch::optim::StepLR scheduler(optimizer, /*step_size=*/5, /*gamma=*/0.9);
    
    torch::nn::MSELoss criterion;
    
    // convert to tensor
    torch::Tensor train_input_tensor = input_data_to_tensor(train_input).to(device_);
    torch::Tensor train_target_tensor = target_data_to_tensor(train_target).to(device_);
    torch::Tensor val_input_tensor = input_data_to_tensor(val_input).to(device_);
    torch::Tensor val_target_tensor = target_data_to_tensor(val_target).to(device_);
    
    // train loop
    std::cout << "️Start training loop..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (int epoch = 0; epoch < num_epochs_; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // train phase
        model_->train();
        double train_loss = 0.0;
        int train_batches = 0;
        
        auto batches = data_loader(train_input_tensor, train_target_tensor, batch_size_);
        const int num_batches = batches.size();
        const int train_batch_count = static_cast<int>(num_batches * 0.8);

        // get normalization parameters
        torch::Tensor target_mean_tensor = model_->named_buffers()["target_mean"];
        torch::Tensor target_std_tensor = model_->named_buffers()["target_std"];
        
        for (int i = 0; i < train_batch_count; ++i) {
            optimizer.zero_grad();
            torch::Tensor predictions = model_->forward(batches[i].data_);
            // normalization loss
            torch::Tensor predictions_norm = (predictions - target_mean_tensor) / target_std_tensor;
            torch::Tensor targets_norm = (batches[i].target_ - target_mean_tensor) / target_std_tensor;
            torch::Tensor loss = criterion(predictions_norm, targets_norm);
            
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
            optimizer.step();
            
            train_loss += loss.item<float>();
            train_batches++;
        }
        
        scheduler.step();
        
        // validation phase
        model_->eval();
        double val_loss = 0.0;
        double val_mae = 0.0;
        int val_batches = 0;
        
        torch::NoGradGuard no_grad;
        for (int i = train_batch_count; i < num_batches; ++i) {
            torch::Tensor predictions = model_->forward(batches[i].data_);
            // normalization loss
            torch::Tensor predictions_norm = (predictions - target_mean_tensor) / target_std_tensor;
            torch::Tensor targets_norm = (batches[i].target_ - target_mean_tensor) / target_std_tensor;
            torch::Tensor loss = criterion(predictions_norm, targets_norm);
            
            val_loss += loss.item<float>();
            val_mae += torch::mean(torch::abs(predictions - batches[i].target_)).item<float>();
            val_batches++;
        }
        
        double avg_train_loss = train_loss / train_batches;
        double avg_val_loss = val_loss / val_batches;
        double avg_val_mae = val_mae / val_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
        
        // if ((epoch + 1) % 5 == 0 || epoch == 0) {
        std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << num_epochs_ << ": "
                    << "Loss=" << std::fixed << std::setprecision(6) << avg_train_loss << ", "
                    << "Val_Loss=" << avg_val_loss << ", "
                    << "MAE=" << avg_val_mae << ", "
                    << "Time=" << epoch_duration.count() / 1000.0 << "s" << std::endl;
        // }
    }
    
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Training completed!" << std::endl;
    std::cout << "Total time: " << total_duration.count() / 1000.0 << "s" << std::endl;
}

void PressureSolverTrainer::cal_python_train(const std::vector<nnInputData>& input_data, const std::vector<float>& target_data){
    setenv("PYTHONHOME", "/home/letian/anaconda3/envs/cuda_acceleration", 1);
    setenv("PYTHONPATH", "/home/letian/anaconda3/envs/cuda_acceleration/lib/python3.10/site-packages", 1); 

    py::scoped_interpreter guard{}; // execute python interpreter

    // 1. add py_solver to sys.path, let python use default module search path
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "py_solver");

    // 2. convert input_data to numpy array
    size_t N = input_data.size();
    std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(N), 5};
    py::array_t<float> input_np(shape);
    auto buf = input_np.mutable_unchecked<2>();
    for (size_t i = 0; i < N; ++i) {
        buf(i, 0) = input_data[i].numberDensity_;
        buf(i, 1) = input_data[i].divU_;
        buf(i, 2) = input_data[i].APi_;
        buf(i, 3) = input_data[i].APjSum_;
        buf(i, 4) = input_data[i].ghostWeightP_;
    }

    // 3. convert target_data to numpy array (ensure it is a 2D array)
    std::vector<py::ssize_t> target_shape = {static_cast<py::ssize_t>(target_data.size()), 1};
    py::array_t<float> target_np(target_shape);
    auto target_buf = target_np.mutable_unchecked<2>();
    for (size_t i = 0; i < target_data.size(); ++i) {
        target_buf(i, 0) = target_data[i];
    }

    // 4. import Python module and class
    py::object model_module = py::module_::import("pressure_solver_model_with_norm");
    py::object TrainerClass = model_module.attr("OptimizedPressureSolverTrainerWithNorm");

    // 5. instantiate Trainer object (can pass parameters, or use default parameters)
    py::object trainer = TrainerClass(
        py::arg("batch_size")=20480,
        py::arg("num_epochs")=20,
        py::arg("compile_model")=true // you can adjust parameters as needed
    );

    // 6. call train_data method
    trainer.attr("train_data")(input_np, target_np, true);

    std::cout << "Python training completed!" << std::endl;
}

// save model
void PressureSolverTrainer::save_model(const std::string& filename, const std::string& type) {
    if (!model_) {
        throw std::runtime_error("Model not trained");
    }
    
    model_->eval();
    
    if (type == "pt") {
        std::string pt_filename = filename + ".pt";
        torch::save(model_, pt_filename);
        std::cout << "Model saved: " << pt_filename << std::endl;
    } else if (type == "onnx") {
        std::cout << "ONNX export is not supported yet in libtorch" << std::endl;
        std::string pt_filename = filename + ".pt";
        torch::save(model_, pt_filename);
        std::cout << "model is saved as pt format: " << pt_filename << std::endl;
    } else {
        throw std::runtime_error("Unsupported model format: " + type);
    }
}

// load model
void PressureSolverTrainer::load_model(const std::string& filename) {
    std::string pt_filename = filename + ".pt";
    if (!std::filesystem::exists(pt_filename)) {
        throw std::runtime_error("Model file not found: " + pt_filename);
    }
    
    model_ = std::make_shared<DNNWithNorm>(5, base_neurons_, dropout_prob_);
    torch::load(model_, pt_filename);
    model_->eval();
    
    // load normalization parameters
    load_normalization_params();
    
    std::cout << "Model loaded: " << pt_filename << std::endl;
}

// predict single sample
SC PressureSolverTrainer::predict_single(const nnInputData& input_data) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }   
    
    model_->eval();
    torch::NoGradGuard no_grad;
    
    torch::Tensor input_tensor = torch::zeros({1, 5}, torch::kFloat32).to(device_);
    input_tensor[0][0] = input_data.numberDensity_;
    input_tensor[0][1] = input_data.divU_;
    input_tensor[0][2] = input_data.APi_;
    input_tensor[0][3] = input_data.APjSum_;
    input_tensor[0][4] = input_data.ghostWeightP_;
    
    torch::Tensor prediction = model_->forward(input_tensor);
    return prediction.item<float>();
}

// batch predict
void PressureSolverTrainer::predict_batch(const std::vector<nnInputData>& input_data, std::vector<SC>& output_data) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }
    
    model_->eval();
    torch::NoGradGuard no_grad;
    
    int num_samples = input_data.size();
    torch::Tensor input_tensor = input_data_to_tensor(input_data).to(device_);
    
    torch::Tensor predictions = model_->forward(input_tensor);
    
    output_data.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        output_data[i] = predictions[i][0].item<float>();
    }
}

// predict (compatible with original interface)
torch::Tensor PressureSolverTrainer::predict(const torch::Tensor& input_data) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }   
    
    model_->eval();
    torch::NoGradGuard no_grad;
    
    torch::Tensor input_tensor = input_data.to(device_);
    if (input_tensor.dim() == 1) {
        input_tensor = input_tensor.unsqueeze(0);
    }
    
    return model_->forward(input_tensor);
}

// predict from CSV file
std::vector<SC> PressureSolverTrainer::predict_from_csv(const std::vector<std::string>& csv_files) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }
    
    std::cout << "Predict from CSV file..." << std::endl;
    
    // load data but not normalize (because model will handle it)
    std::vector<nnInputData> all_input_data;
    std::vector<SC> all_target_data;
    
    for (const auto& csv_file : csv_files) {
        if (!std::filesystem::exists(csv_file)) {
            throw std::runtime_error("File not found: " + csv_file);
        }
        
        TrainingData file_data = load_single_csv(csv_file);
        all_input_data.insert(all_input_data.end(), file_data.input_data.begin(), file_data.input_data.end());
        all_target_data.insert(all_target_data.end(), file_data.target_data.begin(), file_data.target_data.end());
    }
    
    // batch predict
    std::vector<SC> predictions;
    predict_batch(all_input_data, predictions);
    
    std::cout << "Prediction completed, processed " << predictions.size() << " samples" << std::endl;
    
    return predictions;
}

// variadic template: supports any number of string arguments for prediction
template<typename... Args>
std::vector<SC> PressureSolverTrainer::predict_from_csv(const std::string& first_file, Args... args) {
    std::vector<std::string> files = {first_file, args...};
    return predict_from_csv(files);
}

// overload: supports single CSV file
std::vector<SC> PressureSolverTrainer::predict_from_csv(const std::string& csv_file) {
    return predict_from_csv(std::vector<std::string>{csv_file});
}

void execute_in_C_plus_plus(){
    try {
        std::cout << "Pressure solver trainer test" << std::endl;
        
        // create trainer
        PressureSolverTrainer trainer(
            0.001,  // learning_rate
            0.001,  // weight_decay
            20480,  // batch_size
            20,     // num_epochs (reduce for quick test)
            0.025,  // dropout_prob
            16      // base_neurons
        );
        
        // define multiple training files
        std::vector<std::string> training_files = {
            "train_data/training_poisson_data_RPM500.csv",
            "train_data/training_poisson_data_RPM1000.csv",
            "train_data/training_poisson_data_RPM2500.csv"
        };
        
        // check if files exist
        std::vector<std::string> existing_files;
        for (const auto& file : training_files) {
            if (std::filesystem::exists(file)) {
                existing_files.push_back(file);
            } else {
                std::cout << "️File not found: " << file << std::endl;
            }
        }
        
        if (existing_files.empty()) {
            std::cout << "No training files found, using default file" << std::endl;
            existing_files = {"train_data/training_poisson_data_RPM500.csv"};
        }
        
        std::cout << "Using training files: ";
        for (const auto& file : existing_files) {
            std::cout << file << " ";
        }
        std::cout << std::endl;
        
        // demonstrate variadic template training functionality
        if (existing_files.size() >= 3) {
            // use variadic template
            std::cout << "\nUsing variadic template for training..." << std::endl;
            trainer.train(existing_files[0], existing_files[1], existing_files[2]);
        } else {
            // use single file
            trainer.train(existing_files[0]);
        }
        
        // save model
        trainer.save_model("model/model-best-with-norm", "pt");
        
        // test single sample prediction
        nnInputData test_input = {1.0, 0.5, 0.3, 0.2, 0.1};
        SC prediction = trainer.predict_single(test_input);
        std::cout << "Single sample prediction result: " << prediction << std::endl;
        
        // test batch prediction
        std::vector<nnInputData> batch_input = {
            {1.0, 0.5, 0.3, 0.2, 0.1},
            {2.0, 0.8, 0.4, 0.3, 0.2}
        };
        std::vector<SC> batch_output;
        trainer.predict_batch(batch_input, batch_output);
        
        std::cout << "Batch prediction result: ";
        for (const auto& pred : batch_output) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;


        std::vector<std::string> test_files = {"test_data/test_poisson_data.csv"};
        // demonstrate variadic template prediction functionality
        std::cout << "\nTest prediction from CSV file..." << std::endl;
        try {
            // use variadic template for prediction
            std::vector<SC> csv_predictions;
            if (test_files.size() >= 2) {
                std::cout << "use variadic template to predict multiple files..." << std::endl;
                csv_predictions = trainer.predict_from_csv(test_files[0], test_files[1]);
            } else {
                csv_predictions = trainer.predict_from_csv(test_files[0]);
            }
            
            std::cout << "CSV prediction result statistics:" << std::endl;
            std::cout << "   number of samples: " << csv_predictions.size() << std::endl;
            if (!csv_predictions.empty()) {
                auto min_pred = *std::min_element(csv_predictions.begin(), csv_predictions.end());
                auto max_pred = *std::max_element(csv_predictions.begin(), csv_predictions.end());
                auto avg_pred = std::accumulate(csv_predictions.begin(), csv_predictions.end(), 0.0) / csv_predictions.size();
                std::cout << "   min value: " << min_pred << std::endl;
                std::cout << "   max value: " << max_pred << std::endl;
                std::cout << "   average value: " << avg_pred << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "️CSV prediction test failed: " << e.what() << std::endl;
        }
        
        std::cout << "\nAll operations completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
}

void execute_in_python(){
    std::vector<std::string> training_files = {
            "train_data/training_poisson_data_RPM500.csv",
            "train_data/training_poisson_data_RPM1000.csv",
            "train_data/training_poisson_data_RPM2500.csv"
    };

    PressureSolverTrainer trainer(
        0.001,  // learning_rate
        0.001,  // weight_decay
        20480,  // batch_size
        20,     // num_epochs (reduce for quick test)
        0.025,  // dropout_prob
        16      // base_neurons
    );

    TrainingData data = trainer.load_csv_data(training_files);
    trainer.cal_python_train(data.input_data, data.target_data);
    
}

int main() {
    execute_in_python();
    return 0;
}

