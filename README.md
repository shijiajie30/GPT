# GPT
simplified GPT
## 项目概述
本项目实现了一个简化版本的 GPT 模型，用于在 WikiText-2 数据集上进行语言建模任务。模型基于 Transformer 架构，包含词嵌入层、位置编码层、多个 Transformer 块以及一个解码器层。通过训练，该模型能够对输入的文本序列进行预测，并计算相应的困惑度等指标。


## 项目结构
.
├── dataset.py        # 数据集加载和处理模块
├── main.py           # 主程序入口
├── model.py          # 模型定义模块
├── train_test.py     # 训练和测试函数模块
└── README.md         # 项目说明文档
