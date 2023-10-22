# CorNER: Comprehensive Bug Report Analysis (CorNER：全面的错误报告分析)

## Installation (安装)

Before running the scripts, you need to install several dependencies as listed below. (在运行脚本之前，您需要安装以下列出的几个依赖项。)

```plaintext
bert-serving-client
bert-serving-server
...
```

## Module Description (模块描述)

### 1. CNN Module (CNN 模块)

- **Data Cleaning (数据清洗)**: `load_data.py`
- **Dataset Construction (构建数据集)**: `mydatasets.py`
- **TextCNN Model (textcnn 模型)**: `model.py`
- **Training and Evaluation (训练和评估)**: `train.py`
- **CNN Configuration and Execution (CNN 参数配置和执行)**: `cnn.py`

### 2. Named Entity Recognition (NER) Module (命名实体识别模块)

- **Preprocessing (预处理)**: `data_processing.py`
- **NER Training (命名实体识别训练)**: `Method2_RandomForest.py`

### 3. Masked Text + Logistic Regression Module (掩码文本+逻辑回归模块)

- **Masked Text (掩码文本)**: `mask_similarity.py`
- **Logistic Regression (逻辑回归)**: `LR.py`

## Execution Steps (执行步骤)

1. **Decompression (解压缩)**: Unrar all the `.rar` files to extract folders with the same names.

2. **BERT Server Initialization (启动 BERT 服务器)**:

   Start the BERT server with the following command:

   (使用以下命令启动 BERT 服务器：)

   ```sh
   bert-serving-start -pooling_strategy REDUCE_MAX -model_dir ~/xxxx/Dul_Bug/uncased_L-12_H-768_A-12/ -num_worker=2 -max_seq_len=NONE
   ```

3. **Script Execution (脚本执行)**:

   Run `main.py` to integrate and execute all modules. You can test specific parts by uncommenting the respective sections of the code.

   (运行 `main.py` 以集成和执行所有模块。您可以通过取消注释代码的相应部分来测试特定部分。)

