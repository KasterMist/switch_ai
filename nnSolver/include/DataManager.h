#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <vector>
#include <string>
#include <optional>
#include <utility>
#include <chrono>

#define SC float


struct nnInputData {
    SC numberDensity_;
    SC divU_;
    SC APi_;
    SC APjSum_;
    SC ghostWeightP_;
};

struct TrainingData {
    std::vector<nnInputData> inputs;
    std::vector<SC> outputs;
    int num_samples;
};

class DataManager {
public:
    DataManager();
    
    ~DataManager();
    

    SC safe_stof(const std::string& str, const std::string& field_name, int line_num);
    TrainingData load_single_csv(const std::string& csv_file);
    
private:
    
};

#endif // DATAMANAGER_H 