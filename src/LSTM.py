import os
import struct
import numpy as np
import torch
import torch.nn as nn
import argparse
from gensim.models import word2vec
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.functional import F
from tqdm import tqdm
from utils import load_data, train, test

torch.manual_seed(0)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, dropout, device):
        super(LSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=2)
        if dropout == 1:
            self.dropout = nn.Dropout(p=0.7)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :].reshape(self.batch_size, -1)
        if hasattr(self, 'dropout'):
            output = self.dropout(output)
        output = self.fc(output)
        return output
    
    def init_weight(self, type):
        if type == 'uniform':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight.data, -1, 1)
                    if len(m.bias.shape) < 2:
                        nn.init.uniform_(m.bias.data.unsqueeze(0), -1, 1)
                    else:
                        nn.init.uniform_(m.bias.data, -1, 1)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.uniform_(param.data, -1, 1)
                        elif 'weight_hh' in name:
                            nn.init.uniform_(param.data, -1, 1)
                        elif 'bias' in name:
                            if len(param.data.shape) < 2:
                                nn.init.uniform_(param.data.unsqueeze(0), -1, 1)
                            else:
                                nn.init.uniform_(param.data, -1, 1)
        elif type == 'normal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        nn.init.normal_(m.bias.data.unsqueeze(0))
                    else:
                        nn.init.normal_(m.bias.data)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.normal_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.normal_(param.data)
                        elif 'bias' in name:
                            if len(param.data.shape) < 2:
                                nn.init.normal_(param.data.unsqueeze(0))
                            else:
                                nn.init.normal_(param.data)
        elif type == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        nn.init.xavier_uniform_(m.bias.data.unsqueeze(0))
                    else:
                        nn.init.xavier_uniform_(m.bias.data)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            if len(param.data.shape) < 2:
                                nn.init.xavier_uniform_(param.data.unsqueeze(0))
                            else:
                                nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            if len(param.data.shape) < 2:
                                nn.init.xavier_uniform_(param.data.unsqueeze(0))
                            else:
                                nn.init.xavier_uniform_(param.data)
                        elif 'bias' in name:
                            if len(param.data.shape) < 2:
                                nn.init.xavier_uniform_(param.data.unsqueeze(0))
                            else:
                                nn.init.xavier_uniform_(param.data)
        elif type == 'orthogonal':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        nn.init.orthogonal_(m.bias.data.unsqueeze(0))
                    else:
                        nn.init.orthogonal_(m.bias.data)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            if len(param.data.shape) < 2:
                                nn.init.orthogonal_(param.data.unsqueeze(0))
                            else:
                                nn.init.orthogonal_(param.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--word2vec_path', type=str, default='Dataset/wiki_word2vec_50.bin', help='Path to word2vec model')
    parser.add_argument('--train_path', type=str, default='Dataset/train.txt', help='Path to original training data')
    parser.add_argument('--val_path', type=str, default='Dataset/validation.txt', help='Path to original validation data')
    parser.add_argument('--test_path', type=str, default='Dataset/test.txt', help='Path to original test data')
    parser.add_argument('--input_size', type=int, default=50, help='Input feature number for word vector')
    parser.add_argument('--hidden_size', type=int, default=100, help='The feature number for hidden state')
    parser.add_argument('--num_layers', type=int, default=1, help='The number of layers for LSTM')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The learning rate for training')
    parser.add_argument('--epoch', type=int, default=10, help='The epoch for training')
    parser.add_argument('--seq_len', type=int, default=50, help='The length of sequence')
    parser.add_argument('--init_type', type=str, default=None, help='Initialization type for weights in network')
    parser.add_argument('--dropout', type=int, default=0, help='Whether to build a dropout layer and drop out some neurons')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据加载与格式化
    model_word2vec = KeyedVectors.load_word2vec_format(args.word2vec_path,binary=True)
    print("Loading data...")
    train_loader = load_data(args.train_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    val_loader = load_data(args.val_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    test_loader = load_data(args.test_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)

    # 模型训练
    print("Training...")
    model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_size=args.batch_size, dropout=args.dropout, device=device)
    model.init_weight(args.init_type)
    model = model.to(device)
    tr_loss_ls, val_acc_ls = train(train_loader=train_loader, val_loader=val_loader, model=model, model_type='LSTM', seq_len=args.seq_len, batch_size=args.batch_size, epoch=args.epoch, learning_rate=args.learning_rate, device=device)

    # 模型测试
    print("testing...")
    test_model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.num_layers, batch_size=args.batch_size, dropout=args.dropout, device=device).to(device=device)
    test_acc, test_F1, test_loss = test(loader=test_loader, type='test', model_type='LSTM', model=test_model, device=device, batch_size=args.batch_size, seq_len=args.seq_len)
    print(f'Test Accuracy: {test_acc}, Test F1: {test_F1}')