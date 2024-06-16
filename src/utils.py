import os
import struct
import numpy as np
import torch
import matplotlib.pyplot as plt
from gensim.models import word2vec
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

def load_data(f_path, model_word2vec, seq_len, batch_size=64):
    '''
    将文本数据处理为dataloader
    f_path: 原数据集路径，如train.txt、val.txt、test.txt
    seq_len: 每个句子保留的词语个数，不足的补0
    batch_size: 每个batch的大小
    
    return: x: 包含所有句子的三维tensor，每个元素是一个句子的tensor
            y: 包含所有句子的标签
    '''
    with open(f_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        y = []
        x = []
        num = 0
        for line in tqdm(lines):
            line = line.split()
            y.append(int(line[0]))
            sen_each = []
            for word in line[1:]:
                try:
                    sen_each.append(model_word2vec[word])
                except:
                    sen_each.append(np.ones(50))
            if len(sen_each) < seq_len:
                sen_each = np.pad(sen_each, ((0, seq_len-len(sen_each)), (0, 0)), 'constant', constant_values=(0, 0))
            if len(sen_each) > seq_len:
                sen_each = sen_each[:seq_len]
            sen_each = torch.tensor(np.array(sen_each))
            x.append(sen_each)
    x = torch.tensor(np.array(x, dtype=np.float32))
    y = torch.tensor(np.array(y), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def train(train_loader, val_loader, model, model_type, seq_len, batch_size, epoch, learning_rate, device):
    '''
    train_loader: 训练集的dataloader
    val_loader: 验证集的dataloader
    model: 深度学习模型
    model_type: 模型类型，如TextCNN, LSTM, MLP等
    word_len: 每个句子保留的词语个数
    batch_size: 每个batch的大小
    epoch: 训练的轮数
    learning_rate: 学习率
    device: 训练设备

    return: model: 训练好的模型
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()
    tr_loss_ls = []
    val_acc_ls = []
    ori_val_loss = val_loss = -1
    for i in range(epoch):
        model.train()
        total = 0
        tr_loss = 0
        running_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            if model_type == 'TextCNN':
                x = x.reshape(batch_size, 1, seq_len, -1)
            elif model_type == 'LSTM':
                x = x.reshape(batch_size, seq_len, -1)
            elif model_type == 'MLP':
                x = x.reshape(batch_size, -1)
            else:
                raise ValueError('model_type should be TextCNN, LSTM or MLP')
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += y.size(0)
            running_loss += loss.item()
            tr_loss += loss.item() * batch_size
            if batch_idx % 50 == 49:
                print('epoch: {}, batch: {}, training loss: {}'.format(i, batch_idx, running_loss/50))
                running_loss = 0
        tr_loss_ls.append(tr_loss / total)
        ori_val_loss = val_loss
        val_acc, val_F1, val_loss = test(loader=val_loader, type='val', model_type=model_type, model=model, device=device, batch_size=batch_size, seq_len=seq_len)
        # 如果验证集的loss增加了3%，则停止训练，也可固定迭代次数
        # if ori_val_loss != -1 and val_loss > ori_val_loss * 1.03:
        #     break
        val_acc_ls.append(val_acc)
        print('epoch: {}, validation accuracy: {}, validation F1 score: {}, validation loss: {}'.format(i, val_acc, val_F1, val_loss))
    # 保存模型
    torch.save({'model': model.state_dict()}, f'{model_type}.pth')
    return tr_loss_ls, val_acc_ls

def TP_FP_FN(y, predicted):
    TP, FP, FN = 0, 0, 0
    for i in range(y.size(0)):
        if y[i] == 1 and predicted[i] == 1:
            TP += 1
        elif y[i] == 0 and predicted[i] == 1:
            FP += 1
        elif y[i] == 1 and predicted[i] == 0:
            FN += 1
    return TP, FP, FN

def test(loader, type, model_type, model, device, batch_size, seq_len):
    '''
    loader: dataloader
    type: 标明是训练集、验证集还是测试集
    model_type: 模型类型，如TextCNN, LSTM, MLP等
    model: 深度学习模型
    device: 训练设备
    batch_size: 每个batch的大小
    seq_len: 每个句子保留的词语个数
    '''
    model.eval()
    if type == 'val' or type == 'train':
        pass
    elif type == 'test':
        state_dict = torch.load(f'{model_type}.pth')
        model.load_state_dict(state_dict['model'])
    else:
        raise ValueError('type should be val or test')
    correct = 0
    total = 0
    loss = 0
    TP, FP, FN = 0, 0, 0
    with torch.no_grad():
        loss_func = torch.nn.CrossEntropyLoss()
        for batch_idx, (x, y) in enumerate(loader, 0):
            if model_type == 'TextCNN':
                x = x.reshape(batch_size, 1, seq_len, -1)
            elif model_type == 'LSTM':
                x = x.reshape(batch_size, seq_len, -1)
            elif model_type == 'MLP':
                x = x.reshape(batch_size, -1)
            else:
                raise ValueError('model_type should be TextCNN, LSTM or MLP')
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            loss += loss_func(output, y).item() * batch_size
            correct += (predicted == y).sum().item()
            TP += TP_FP_FN(y, predicted)[0]
            FP += TP_FP_FN(y, predicted)[1]
            FN += TP_FP_FN(y, predicted)[2]
    try:
        precision = TP / (TP + FP)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    try:
        F1 = 2 * precision * recall / (precision + recall)
    except:
        F1 = 0
    return correct / total, F1, loss / total
    
    


