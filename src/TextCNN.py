import os
import struct
import numpy as np
import torch
import argparse
from gensim.models import word2vec
from gensim.models import KeyedVectors
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.functional import F
from tqdm import tqdm
from utils import load_data, train, test

torch.manual_seed(0)
np.random.seed(0)

class TextCNN(torch.nn.Module):
    def __init__(self, kernel_num, kernel_size, pooling_size, dropout, seq_len):
        super(TextCNN, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(self.kernel_size[0], 50))
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(self.kernel_size[1], 50))
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=self.kernel_num, kernel_size=(self.kernel_size[2], 50))
        self.pooling = torch.nn.MaxPool2d(kernel_size=self.pooling_size, stride=self.pooling_size)
        in_features = int((seq_len-2) / pooling_size[0]) * self.kernel_num + int((seq_len-4) / pooling_size[0]) * self.kernel_num + int((seq_len-6) / pooling_size[0]) * self.kernel_num
        self.fc = torch.nn.Linear(in_features=in_features, out_features=2)
        if dropout == 1:
            self.dropout = torch.nn.Dropout(p=0.7)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x1 = self.pooling(x1)
        x2 = self.pooling(x2)
        x3 = self.pooling(x3)
        x = torch.cat((x1, x2, x3), dim=2)
        x = x.view(batch_size, -1)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def init_weights(self, type):
        '''
        model: 模型
        type: 初始化类型，如全1、均匀分布、正态分布、xavier、kaiming
        '''
        if type == 'uniform':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.uniform_(m.weight.data, -1, 1)
                    if len(m.bias.shape) < 2:
                        torch.nn.init.uniform_(m.bias.data.unsqueeze(0), -1, 1)
                    else:
                        torch.nn.init.uniform_(m.bias.data, -1, 1)
        elif type == 'normal':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        torch.nn.init.normal_(m.bias.data.unsqueeze(0))
                    else:
                        torch.nn.init.normal_(m.bias.data)
        elif type == 'xavier':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        torch.nn.init.xavier_uniform_(m.bias.data.unsqueeze(0))
                    else:
                        torch.nn.init.xavier_uniform_(m.bias.data)
        elif type == 'orthogonal':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight.data)
                    if len(m.bias.shape) < 2:
                        torch.nn.init.orthogonal_(m.bias.data.unsqueeze(0))
                    else:
                        torch.nn.init.orthogonal_(m.bias.data)
        else:
            print("Choosing default initialization method")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TextCNN')
    parser.add_argument('--word2vec_path', type=str, default='Dataset/wiki_word2vec_50.bin', help='Path to word2vec model')
    parser.add_argument('--train_path', type=str, default='Dataset/train.txt', help='Path to original training data')
    parser.add_argument('--val_path', type=str, default='Dataset/validation.txt', help='Path to original validation data')
    parser.add_argument('--test_path', type=str, default='Dataset/test.txt', help='Path to original test data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=50, help='Length of sequence')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--pooling_size', type=int, default=4, help='Stride for max pooling layer')
    parser.add_argument('--init_type', type=str, default='None', help='Initialization type for weights in network')
    parser.add_argument('--dropout', type=int, default=0, help='Whether to build a dropout layer and drop out some neurons')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_word2vec = KeyedVectors.load_word2vec_format(args.word2vec_path,binary=True)
    print("Loading data...")
    train_loader = load_data(args.train_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    val_loader = load_data(args.val_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    test_loader = load_data(args.test_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)

    print("Training...")
    model = TextCNN(2, [3, 5, 7], (args.pooling_size, 1), args.dropout, args.seq_len).to(device)
    model.init_weights(args.init_type)
    tr_loss_ls, val_acc_ls = train(train_loader=train_loader, val_loader=val_loader, model=model, model_type="TextCNN", seq_len=args.seq_len, batch_size=args.batch_size, epoch=args.epoch, learning_rate=args.learning_rate, device=device)

    print("Testing...")
    test_model = TextCNN(2, [3, 5, 7], (args.pooling_size, 1), args.dropout, args.seq_len).to(device)
    Accuracy, F1, test_loss = test(loader=test_loader, type='test', model_type="TextCNN", model=test_model, device=device, batch_size=args.batch_size, seq_len=args.seq_len)
    print('Accuracy of the network on the test set: %f %%' % (100 * Accuracy))
    print('F1 score of the test set: {}'.format(F1))