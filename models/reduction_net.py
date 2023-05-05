from torch import TensorType, nn
from torch.utils.data import DataLoader, Dataset
from math import floor
from torchmetrics import Accuracy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import argparse

from sklearn import cluster

class Layer():
    def __init__(self,
                 in_dim: int,
                 out_dim: int):
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def __call__(self, x: TensorType) -> TensorType:
        y = self.linear(x)
        y = self.norm(y)
        return y

# class Block():
#     def __init__(self,
#
#                  )

class ReductionNet(nn.Module):
    def __init__(self,
                 input_features: int,   # number of features being inputted
                 layers: list[int],     # list of hidden layer sizes
                 num_classes: int,      # how many classes are in the network in total
                 activation: str,
                 ):
        super().__init__()
        layer_dims = [input_features] + layers
 
        self.layers = [
                Layer(layer_dims[i-1], layer_dims[i])
                for i in range(1, len(layer_dims))
                ]

        for layer in self.layers:
            print(f'{layer.linear = }')

        if   activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        
        self.classifier = nn.Linear(layer_dims[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x: TensorType) -> TensorType:
        for layer in self.layers:
            # print(f'{layer.linear = }')
            # print(f'{x.shape = }')
            # print(f'{x}')
            x = self.activation(layer(x))
        return x

    @torch.no_grad()
    def embed(self, x: TensorType) -> TensorType:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def train_model(self, 
                    train_loader,
                    dev_loader, 
                    classes,
                    epochs,
                    lr,
                    optimizer,
                    ):

        criterion = nn.CrossEntropyLoss()

        if  optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(self.parameters(), lr=lr)
        elif optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr) 
        elif optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)


        accuracy = Accuracy(task='multiclass', num_classes=classes)
        
        for epoch in range(epochs):
            print(f'{epoch = }')
            epoch_loss = 0.0

            counter = 0
            limit = 1000
            for _, (mb_x, mb_y) in enumerate(train_loader):
                
                # print(f'{mb_x = }')

                embeddings = self.forward(mb_x)
                pred_logits = self.classifier(embeddings)

                # print(f'{y_pred = }')
                # print(f'{mb_y = }')

                loss = criterion(pred_logits, mb_y)
                

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if counter == limit:
                    pred = self.softmax(pred_logits).argmax(axis=1)
                    true = mb_y.argmax(axis=1)
                    print(f'{pred = }')
                    print(f'{true = }')
                counter = counter + 1 % limit

            epoch_loss = epoch_loss / len(train_loader)
            print(f'{epoch_loss = }')

            # Eval the model's current performance
            self.eval()
            acc = 0
            total = 0
            for _, (d_mb_x,d_mb_y) in enumerate(dev_loader):
                embeddings = self.forward(d_mb_x)
                y_pred_probabilities = self.softmax(self.classifier(embeddings))
                 
                # true_pos += torch.sum(dev_predicted.argmax(axis=1) == d_mb_y.argmax(axis=1))
                acc += accuracy(y_pred_probabilities.argmax(axis=1), d_mb_y.argmax(axis=1))
                total += 1
            dev_acc = (acc / total)
            print("dev acc = %.3f" % (dev_acc*100))
            self.train()


class TweetDataset(Dataset):
    def __init__(self, dataframe, classes, embedding_column, label_column):
        """
        path: Path to pandas dataframe pickle
        """
        self.df = dataframe.sample(frac=1)
        self.classes = classes

    def __getitem__(self, index):
        row = self.df.iloc[index]

        embed = row[embedding_column]
        label = torch.zeros(self.classes)
        label[row[label_column]] = 1.0

        return embed, label

    def __len__(self):
        return len(self.df)

def main():

    # Define arguments
    parser = argparse.ArgumentParser()
    # Model details
    parser.add_argument('--input_features', type=int, default=300)
    parser.add_argument('--layers', type=str, default='64')
    parser.add_argument('--classes', type=int, default=6)
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--optimizer', type=str, default='adam')
    # Data/Model paths
    parser.add_argument('--save_path', type=str, default='models/reduction_net')
    parser.add_argument('--embedding_col', type=str, default='fasttext')
    parser.add_argument('--label_col', type=str, default='fasttext label')
    parser.add_argument('--data_path', type=str, default='data/dataframe')


    args = parser.parse_args()

    # Load data
    df = pd.read_pickle(args.data_path)
    
    split = floor(len(df) * 0.9)
    
    train_set = TweetDataset(
            dataframe=df.iloc[:split],
            classes=args.classes,
            embedding_column=args.embedding_col,
            label_column=args.label_col
            )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    dev_set = TweetDataset(
            dataframe=df.iloc[split:],
            classes=args.classes,
            embedding_column=args.embedding_col,
            label_column=args.label_col
            )
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)

    # Layer arg format: 3x128,3x64 = block of 3 128, block of 3 64
    layers = [
                int(layer) for layer in args.layers.split(',')
            ]
    print(f'{layers = }')

    model = ReductionNet(
            input_features=args.input_features,
            layers=layers,
            num_classes=args.classes,
            activation=args.activation
            )

    model.train_model(
            train_loader=train_loader, 
            dev_loader=dev_loader, 
            epochs=args.epochs,
            lr=args.lr,
            classes=args.classes,
            optimizer=args.optimizer
            )

    torch.save(model.state_dict, args.save_path)

    # Test cluster performance
    # reduced = df['embedding'].apply(lambda e : model.embed(torch.from_numpy(e)))
    reduced = torch.vstack([model.embed(torch.from_numpy(e)) for e in df[args.embedding_col]])
    decomp = PCA(2).fit_transform(reduced)
 
    plt.scatter(decomp[:,0], decomp[:,1], c=df['label'])
    plt.savefig('pca_bert_clusters')   

    # decomp = TSNE(2).fit_transform(reduced)
    #
    # plt.scatter(decomp[:,0], decomp[:,1], c=df['label'])
    # plt.savefig('tsne_clusters')

if __name__ == '__main__':
    main()



