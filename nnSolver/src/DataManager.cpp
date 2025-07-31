#include "DataManager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <cctype>

DataManager::DataManager() {
    // 构造函数
}

DataManager::~DataManager() {
    // 析构函数
}

// Load single CSV file
// Helper function: clean string and convert to float (supports scientific notation)
SC DataManager::safe_stof(const std::string& str, const std::string& field_name, int line_num) {
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

TrainingData DataManager::load_single_csv(const std::string& csv_file) {
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
            // if (target >= -3000 && target <= 5000) {
            input_data.push_back(input);
            target_data.push_back(target);
            // }
            
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
