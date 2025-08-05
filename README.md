# AI模型训练与推理项目

## 项目概述

这是一个C++和Python混合的AI模型项目，主要分为两个部分：

- **Python端 (py_solver)**: 负责AI模型训练和生成（如ONNX模型）
- **C++端**: 负责模型推理，不能包含机器学习训练相关包（如PyTorch），但可以使用推理相关包（如ONNX Runtime、TVM等）

### 项目特点

- **灵活的模型架构**: 支持多种神经网络类型（DNN、Transformer、CNN等）
- **完整的训练流水线**: 从数据加载到模型训练、保存的完整流程
- **多格式支持**: 支持PyTorch和ONNX格式的模型保存和加载
- **标准化管理**: 支持多种数据标准化方法（标准化、最小-最大归一化）
- **版本控制**: 内置模型版本管理和元数据跟踪
- **推理优化**: 支持批量推理和ONNX运行时优化

## py_solver 模块详解

### 核心文件功能

#### 1. `config.py` - 配置管理
- **功能**: 定义模型配置类和数据标准化信息类
- **主要组件**:
  - `ModelType`: 模型类型枚举（FLEXIBLE_DNN、TRANSFORMER、CNN）
  - `StandardizationInfo`: 标准化参数信息类
  - `MinMaxNormInfo`: 最小-最大归一化参数信息类
  - `model_config`: 主配置类，包含输入维度、输出维度、神经元数量、dropout率等参数
- **特点**: 支持多种激活函数（exp_decay、relu、tanh、sigmoid等）和标准化方法

#### 2. `dnn.py` - 神经网络模型定义
- **功能**: 实现灵活的深度神经网络模型
- **主要组件**:
  - `ExpDecayActivation`: 自定义激活函数 x * exp(-x²/(2*e))
  - `FlexibleDNN`: 灵活的DNN模型，支持多种标准化和激活函数
  - `DNN`: 向后兼容的DNN类
- **特点**: 
  - 支持内置标准化/反标准化
  - 支持ONNX模型导出
  - 支持模型保存和加载
  - 可配置的网络层结构

#### 3. `model_factory.py` - 模型工厂
- **功能**: 提供模型创建的统一接口
- **主要组件**:
  - `create_model()`: 根据配置创建模型实例
  - `create_dnn_model()`: 创建DNN模型（向后兼容）
  - `MODEL_REGISTRY`: 模型类型注册表
- **特点**: 支持扩展新的模型类型，便于添加Transformer、CNN等模型

#### 4. `data_manager.py` - 数据管理
- **功能**: 处理CSV数据加载、验证和预处理
- **主要组件**:
  - `DataManager`: 数据管理器类
  - 支持多文件CSV加载
  - 数据维度验证
  - 训练/验证数据分割
  - 标准化参数计算
- **特点**:
  - 支持通配符文件匹配
  - 自动计算标准化参数
  - 数据摘要和统计信息

#### 5. `model_manager.py` - 模型管理
- **功能**: 处理模型保存、加载和版本管理
- **主要组件**:
  - `ModelManager`: 模型管理器类
  - 模型注册表管理
  - 元数据保存和加载
  - ONNX格式导出
- **特点**:
  - 支持在线和离线训练模式
  - 时间戳版本控制
  - 完整的元数据跟踪
  - 模型列表和删除功能

#### 6. `pipeline_manager.py` - 训练流水线
- **功能**: 提供完整的模型训练和评估流水线
- **主要组件**:
  - `PipelineManager`: 流水线管理器类
  - 数据加载和预处理
  - 模型训练和验证
  - 早停机制
  - 模型评估
- **特点**:
  - 支持在线和离线训练
  - 自动设备选择（CPU/GPU）
  - 训练历史记录
  - 批量训练和保存

#### 7. `inference.py` - 推理引擎
- **功能**: 提供模型推理和预测服务
- **主要组件**:
  - `InferenceEngine`: 推理引擎类
  - `PredictionService`: 预测服务类
  - 支持PyTorch和ONNX模型
  - 批量推理功能
- **特点**:
  - 自动格式检测（.pt/.onnx）
  - 批量预测优化
  - CSV文件预测
  - 元数据返回

#### 8. `launch.py` - 启动脚本
- **功能**: 提供C++调用Python训练的接口
- **主要组件**:
  - `cplusplus_launch_python_training()`: C++调用接口
  - `launch_python_training_in_terminal()`: 终端启动接口
- **特点**: 为C++端提供简单的训练启动接口

### 使用流程

#### 1. 离线训练流程
```python
from py_solver.config import model_config
from py_solver.pipeline_manager import PipelineManager

# 创建配置
config = model_config(
    input_dim=5,
    output_dim=1,
    base_neurons=16,
    norm_type="standardization",
    activation="exp_decay"
)

# 创建流水线
pipeline = PipelineManager(config)

# 训练并保存模型
result = pipeline.train_and_save_offline(
    csv_files="data/*.csv",
    epochs=100,
    model_name="my_model"
)
```

#### 2. 在线训练流程
```python
# 使用numpy数组进行在线训练
result = pipeline.train_and_save_online(
    inputs=input_array,
    outputs=output_array,
    epochs=50,
    model_name="online_model"
)
```

#### 3. 推理流程
```python
from py_solver.inference import InferenceEngine

# 加载模型
engine = InferenceEngine("models/my_model/model.pt")

# 进行预测
predictions = engine.predict(input_data)
```

### 项目特典

1. **模块化设计**: 每个模块职责清晰，便于维护和扩展
2. **配置驱动**: 通过配置文件控制模型行为，无需修改代码
3. **多格式支持**: 同时支持PyTorch和ONNX格式
4. **版本管理**: 内置模型版本控制，便于模型迭代
5. **性能优化**: 支持GPU加速和批量推理
6. **易于集成**: 提供简单的C++调用接口

### 开发规范

- **C++依赖限制**: 禁止包含机器学习训练相关包，允许使用推理相关包
- **测试要求**: 每个功能都有对应的测试用例
- **代码质量**: 简洁、可读、符合最佳实践
- **文档要求**: 关键代码需要英文注释 