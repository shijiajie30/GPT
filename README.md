# GPT
simplified GPT
## 项目概述
本项目实现了一个简化版本的 GPT 模型，用于在 WikiText-2 数据集上进行语言建模任务。模型基于 Transformer 架构，包含词嵌入层、位置编码层、多个 Transformer 块以及一个解码器层。通过训练，该模型能够对输入的文本序列进行预测，并计算相应的困惑度等指标。


## 项目结构
- **project-root**
    - **data**
        - wikitext-2
    - **src**
        - main.py
        - model.py
        - dataset.py
        - train_test.py
    - README.md
