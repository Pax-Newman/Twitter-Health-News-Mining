from torch import TensorType, nn
from torch.utils.data import DataLoader, Dataset
from math import floor
from torchmetrics import Accuracy

import torch
import pandas as pd
import numpy as np

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
        

class ReductionNet(nn.Module):
    def __init__(self,
                 input_features: int,   # number of features being inputted
                 layers: list[int],     # list of hidden layer sizes
                 num_classes: int,      # how many classes are in the network in total
                 ):
        super().__init__()
        layer_dims = [input_features] + layers
 
        self.layers = [
                Layer(layer_dims[i-1], layer_dims[i])
                for i in range(1, len(layer_dims))
                ]

        for layer in self.layers:
            print(f'{layer.linear = }')

        self.activation = nn.ReLU()
        
        self.classifier = nn.Linear(layer_dims[-1], num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x: TensorType) -> TensorType:
        for layer in self.layers:
            # print(f'{layer.linear = }')
            # print(f'{x.shape = }')
            # print(f'{x}')
            x = self.activation(layer(x))
        return x

    def train_model(self, train_loader, dev_loader, epochs: int):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())

        accuracy = Accuracy(task='multiclass', num_classes=6)
        
        for epoch in range(epochs):
            print(f'{epoch = }')
            epoch_loss = 0.0

            counter = 0
            limit = 1000
            for _, (mb_x, mb_y) in enumerate(train_loader):
                
                # print(f'{mb_x = }')

                embeddings = self.forward(mb_x)
                y_logits = self.classifier(embeddings)
                y_pred = self.softmax(y_logits)

                # print(f'{y_pred = }')
                # print(f'{mb_y = }')

                loss = criterion(y_logits, mb_y)
                

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if counter == limit:
                    print(y_pred.argmax(axis=1))
                    print(mb_y.argmax(axis=1))
                counter = counter + 1 % limit

            epoch_loss = epoch_loss / len(train_loader)
            print(f'{epoch_loss = }')

            self.eval()
            acc = 0
            total = 0
            for _, (d_mb_x,d_mb_y) in enumerate(dev_loader):
                embeddings = self.forward(d_mb_x)
                y_pred_probabilities = self.softmax(self.classifier(embeddings))
                
                _, dev_predicted = torch.max(y_pred_probabilities, 1)
                
                # true_pos += torch.sum(dev_predicted.argmax(axis=1) == d_mb_y.argmax(axis=1))
                acc += accuracy(y_pred_probabilities.argmax(axis=1), d_mb_y.argmax(axis=1))
                total += 1
            dev_acc = (acc / total)
            print("dev acc = %.3f" % (dev_acc*100))
            self.train()


class TweetDataset(Dataset):
    def __init__(self, dataframe, classes):
        """
        path: Path to pandas dataframe pickle
        """
        self.df = dataframe.sample(frac=1)
        self.classes = classes

    def __getitem__(self, index):
        row = self.df.iloc[index]

        embed = row['embedding']
        label = torch.zeros(self.classes)
        label[row['label']] = 1.0

        return embed, label

    def __len__(self):
        return len(self.df)

def main():
    path = 'data/dataframe'
    dataframe = pd.read_pickle(path)

    classes = 6
    
    split = floor(len(dataframe) * 0.9)

    train_set = TweetDataset(dataframe.iloc[:split], classes)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    dev_set = TweetDataset(dataframe.iloc[split:], classes)
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=True)

    model = ReductionNet(
            input_features=300,
            layers=[64],
            num_classes=classes
            )

    model.train_model(train_loader=train_loader, dev_loader=dev_loader, epochs=100)

    torch.save(model.state_dict, 'models/reductionnet')



if __name__ == '__main__':
    main()



