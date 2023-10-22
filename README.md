### 安装依赖
bert-serving-client           
bert-serving-server 
....
### 模块介绍
##### <1>：cnn模块
* 1.数据清洗：load_data.py
* 2.构建数据集:mydatasets.py
* 3.textcnn模型：model.py
* 4.训练和评估：train.py
* 5.cnn参数配置和执行：cnn.py
##### <2>：命名实体识别模块
* 1.预处理：data_processing.py
* 2.命名实体识别训练：Method2_RandomForest.py
##### <3>：掩码文本+逻辑回归模块
* 1.掩码文本：mask_similarity.py
* 2.逻辑回归：LR.py
### 执行步骤
* 1 打开bert服务器
    ```python 
    bert-serving-start -pooling_strategy  REDUCE_MAX -model_dir     
     ~/xxxx/Dul_Bug/uncased_L-12_H-768_A-12/ -num_worker=2 - 
    max_seq_len=NONE
    
* 2 执行main.py
   这个代码是为了串联所有模块代码，想测试那部分就打开哪部分的代码



