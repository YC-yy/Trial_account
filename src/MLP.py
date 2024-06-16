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

class MLP(nn.Module):
    def __init__(self, seq_len, h_f1, h_f2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=50 * seq_len, out_features=h_f1)
        self.fc2 = nn.Linear(in_features=h_f1, out_features=h_f2)
        self.fc3 = nn.Linear(in_features=h_f2, out_features=2)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = F.relu(self.fc3(h2))
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument('--word2vec_path', type=str, default='Dataset/wiki_word2vec_50.bin', help='Path to word2vec model')
    parser.add_argument('--train_path', type=str, default='Dataset/train.txt', help='Path to original training data')
    parser.add_argument('--val_path', type=str, default='Dataset/validation.txt', help='Path to original validation data')
    parser.add_argument('--test_path', type=str, default='Dataset/test.txt', help='Path to original test data')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the dataloader')
    parser.add_argument('--epoch', type=int, default=10, help='The epoch for training')
    parser.add_argument('--seq_len', type=int, default=50, help='The length of sequence')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate for training')
    parser.add_argument('--h_layer1', type=int, default=1000, help='The number of neurons in the first hidden layer')
    parser.add_argument('--h_layer2', type=int, default=100, help='The number of neurons in the second hidden layer')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_word2vec = KeyedVectors.load_word2vec_format('Dataset/wiki_word2vec_50.bin',binary=True)
    print("Loading data...")
    train_loader = load_data(args.train_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    val_loader = load_data(args.val_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)
    test_loader = load_data(args.test_path, model_word2vec, seq_len=args.seq_len, batch_size=args.batch_size)

    print("Training...")
    model = MLP(seq_len=args.seq_len, h_f1=args.h_layer1, h_f2=args.h_layer2).to(device=device)
    tr_loss_ls, val_acc_ls = train(train_loader=train_loader, val_loader=val_loader, model=model, model_type='MLP', seq_len=args.seq_len, batch_size=args.batch_size, epoch=args.epoch, learning_rate=args.learning_rate, device=device)

    print("Testing...")
    test_model = MLP(seq_len=args.seq_len, h_f1=args.h_layer1, h_f2=args.h_layer2).to(device=device)
    test_acc, test_F1, test_loss = test(loader=test_loader, type='test', model_type='MLP', model=test_model, device=device, batch_size=args.batch_size, seq_len=args.seq_len)
    print(test_acc, test_F1)