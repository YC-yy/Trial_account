# 实验二——情感分析代码说明
本项目包含了情感分析TextCNN, LSTM与MLP实现。运行代码位于`src`文件夹中，数据集位于`Dataset`文件夹中。环境配置请参见`src/requirements.txt`。
## 关键步骤运行方式
### 模型运行
+ TextCNN:
```bash
python src/TextCNN.py
```
可通过`python src/TextCNN.py -h`获取帮助及参数说明。
+ LSTM
```bash
python src/LSTM.py
```
可通过`python src/LSTM.py -h`获取帮助及参数说明。
+ MLP
```bash
python src/MLP.py
```
可通过`python src/MLP.py -h`获取帮助及参数说明。
### 超参数调节
具体细节请参照脚本`src/Tune_para.sh`，以TextCNN `learning_rate`参数为例，代码如下：
```bash
for lr in 0.0005 0.005 0.05 0.5
do
    python src/TextCNN.py --learning_rate $lr
done
```
### Dropout设置
以TextCNN为例，如果要开启Dropout层，运行代码如下：
```bash
python src/TextCNN.py --epoch 30 --dropout 1
```
## 项目结构
其中`Dataset`文件夹中的文件需要额外放置。
```
.
├── Dataset
│   ├── test.txt
│   ├── train.txt
│   ├── validation.txt
│   └── wiki_word2vec_50.bin
├── src
│   ├── Tune_para.sh
│   ├── LSTM.py
│   ├── TextCNN.py
│   ├── MLP.py
│   ├── utils.py
│   └── requirements.txt
└── readme.md
```