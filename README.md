# paddle-AutoInt

## 目录

- [paddle-AutoInt](#paddle-autoint)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 数据集和复现精度](#2-数据集和复现精度)
    - [复现精度](#复现精度)
  - [3. 准备数据与环境](#3-准备数据与环境)
    - [3.1 准备环境](#31-准备环境)
    - [3.2 准备数据](#32-准备数据)
  - [4. 开始使用](#4-开始使用)
    - [4.1 模型训练](#41-模型训练)
    - [4.2 模型预测](#42-模型预测)
  - [5. 模型推理部署](#5-模型推理部署)
  - [6. 自动化测试脚本](#6-自动化测试脚本)
  - [7. LICENSE](#7-license)
  - [8. 参考链接与文献](#8-参考链接与文献)


## 1. 简介

详见[model/rank/autoint/readme.md]()

**论文:** [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)

**参考repo:** [shichence/AutoInt](https://github.com/shichence/AutoInt)

在此非常感谢`shichence`等人贡献的[shichence/AutoInt](https://github.com/shichence/AutoInt)，提高了本repo复现论文的效率。

## 2. 数据集和复现精度

训练及测试数据集选用[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge/)所用的Criteo数据集。该数据集包括两部分：训练集和测试集。训练集包含一段时间内Criteo的部分流量，测试集则对应训练数据后一天的广告点击流量。
每一行数据格式如下所示：
```
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<integer feature>```代表数值特征（连续特征），共有13个连续特征。```<categorical feature>```代表分类特征（离散特征），共有26个离散特征。相邻两个特征用```\t```分隔，缺失特征用空格表示。测试集中```<label>```特征已被移除。

基于上述数据集，我们使用了参考repo及原论文给出的数据预处理方法，把原Criteo数据集转换成如下的数据格式：
```
<label> <feat index 1> ... <feat index 39> <feat value 1> ... <feat value 39>
```
其中```<label>```表示广告是否被点击，点击用1表示，未点击用0表示。```<feat index>```代表特征索引，用于取得连续特征及分类特征对应的embedding。```<feat value>```代表特征值，对于分类特征，```<feat value 14>```至```<feat value 39>```的取值为1，对于连续特征，```<feat value 1>```至```<feat value 13>```的取值为连续数值。相邻两个栏位用空格分隔。测试集中```<label>```特征已被移除。


### 复现精度
|                  |   Epoch     |   Number of Head    |  Interaction Layer    | Deep Layer Size | Hidden Size | Use Wide? | AUC       |
| ---------------  | --------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| AutoInt **(论文)**  |  3 |   2  |  [32,32,32]        |   []        | 16 | N |0.8044
| AutoInt **(论文)**  |  3 |   2  |  [64,64,64]        |   []        | 16 | N |0.8056
| AutoInt **(论文)**  |  3 |   2  |  [64,64,64]        |   [400,400,400]  |   16   | N |0.8065
| paddle-autoint **(本项目)** |  2    |   2   |    [64,64,64]     |     [400,400,400]  |  16  | N  | 0.8070

- 训练权重下载链接：在文件夾```weights```下
- 原论文项目的训练日志在```logs/original_autoint.log```
- 我们的复现达到验收标准```AUC 0.8061```
## 3. 准备数据与环境


### 3.1 准备环境

- 框架：
  - paddlepaddle-gpu >= 2.2.0

运行`pip install paddlepaddle-gpu`即可

- 下载PaddleRec
```
git clone https://github.com/PaddlePaddle/PaddleRec/
cd PaddleRec
```
把本repo的文件夹```datasets```, ```models```和```test_tipc```复制到PaddleRec对应的目录下

### 3.2 准备数据

```
cd datasets/criteo_autoint
sh run.sh
```

## 4. 开始使用

### 4.1 模型训练

- 全量数据训练
```
cd models/rank/autoint
python -u ../../../tools/trainer.py -m config_bigdata.yaml -o runner.seed=2018
```

- 少量数据训练
```
cd models/rank/autoint
python -u ../../../tools/trainer.py -m config.yaml
```
- 部分训练日志在```logs/ours.log```
  

### 4.2 模型预测
- 全量数据
```
cd models/rank/autoint
python -u ../../../tools/infer.py -m config_bigdata.yaml -o runner.seed=2018
```

- 少量数据
```
cd models/rank/autoint
python -u ../../../tools/infer.py -m config.yaml
```

- 预测结果在```logs/infer.log```


## 5. 模型推理部署
暂无

## 6. 自动化测试脚本

- tipc创建指南请见[tipc创建及基本使用方法。](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/tipc/train_infer_python/test_train_infer_python.md)
- 本项目TIPC脚本测试命令详见[Linux GPU/CPU 基础训练推理测试](test_tipc/docs/test_train_inference_python.md)
```bash
#测试环境准备脚本
bash test_tipc/prepare.sh test_tipc/configs/autoint/train_infer_python.txt lite_train_lite_infer
```

```bash
#测试训练验证推理一体化脚本
bash test_tipc/test_train_inference_python.sh test_tipc/configs/autoint/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
[33m Run successfully with command - python -u tools/trainer.py -m ./models/rank/autoint/config.yaml -o runner.print_interval=2 runner.use_gpu=True  runner.model_save_path=./test_tipc/output/norm_train_gpus_0_autocast_False runner.epochs=1   auto_cast=False runner.train_batch_size=2 runner.train_data_dir=../../../test_tipc/data/train   !  [0m
[33m Run successfully with command - python -u test_tipc/configs/autoint/to_static.py -m ./models/rank/autoint/config.yaml -o runner.CE=true runner.model_init_path=./test_tipc/output/norm_train_gpus_0_autocast_False/0 runner.model_save_path=./test_tipc/output/norm_train_gpus_0_autocast_False!  [0m

...

[33m Run successfully with command - python -u tools/paddle_infer.py --model_name=autoint --reader_file=models/rank/autoint/criteo_reader.py --use_gpu=False --enable_mkldnn=True --cpu_threads=6 --model_dir=./test_tipc/output/norm_train_gpus_-1_autocast_False/ --batchsize=10 --data_dir=test_tipc/data/infer --benchmark=True --precision=fp32   > ./test_tipc/output/python_infer_cpu_usemkldnn_True_threads_6_precision_fp32_batchsize_10.log 2>&1 !  [0m
[33m Run successfully with command - python -u tools/paddle_infer.py --model_name=autoint --reader_file=models/rank/autoint/criteo_reader.py --use_gpu=False --enable_mkldnn=False --cpu_threads=1 --model_dir=./test_tipc/output/norm_train_gpus_-1_autocast_False/ --batchsize=10 --data_dir=test_tipc/data/infer --benchmark=True --precision=fp32   > ./test_tipc/output/python_infer_cpu_usemkldnn_False_threads_1_precision_fp32_batchsize_10.log 2>&1 !  [0m
[33m Run successfully with command - python -u tools/paddle_infer.py --model_name=autoint --reader_file=models/rank/autoint/criteo_reader.py --use_gpu=False --enable_mkldnn=False --cpu_threads=6 --model_dir=./test_tipc/output/norm_train_gpus_-1_autocast_False/ --batchsize=10 --data_dir=test_tipc/data/infer --benchmark=True --precision=fp32   > ./test_tipc/output/python_infer_cpu_usemkldnn_False_threads_6_precision_fp32_batchsize_10.log 2>&1 !  [0m
```



## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
**参考论文:** [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921.pdf)

**参考repo:** [shichence/AutoInt](https://github.com/shichence/AutoInt)