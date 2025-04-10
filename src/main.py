import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from model import SimplifiedGPT
from dataset import WikiTextDataset
from train_test import train_model, test_model
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    # 数据路径
    train_data_path = Path(r'D:\AI\deep learning projects\NLP\GPT\data\wikitext-2\wiki.train.tokens')
    valid_data_path = Path(r'D:\AI\deep learning projects\NLP\GPT\data\wikitext-2\wiki.valid.tokens')
    test_data_path = Path(r'D:\AI\deep learning projects\NLP\GPT\data\wikitext-2\wiki.test.tokens')
    # 模型参数
    # 从训练数据集中获取词汇表大小
    train_dataset = WikiTextDataset(train_data_path)
    vocab_size = train_dataset.vocab_size
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    max_seq_length = 512
    dropout = 0.1
    epochs = 10
    batch_size = 16

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练数据集和数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证数据集和数据加载器
    valid_dataset = WikiTextDataset(valid_data_path)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 测试数据集和数据加载器
    test_dataset = WikiTextDataset(test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数和优化器
    model = SimplifiedGPT(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_length, dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device, epochs)

    # 测试模型
    test_model(model, test_dataloader, criterion, device)
