from torch import TensorType, nn
from torch.utils.data import DataLoader, Dataset
import torch

class Layer():
    def __init__(self,
                 in_dim: int,
                 out_dim: int):
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def __call__(self, x: TensorType) -> TensorType:
        y = self.linear(x)
        y = self.norm(x)
        return y
        

class ReductionNet(nn.Module):
    def __init__(self,
                 input_features: int,   # number of features being inputted
                 layers: list[int],     # list of hidden layer sizes
                 num_classes: int,      # how many classes are in the network in total
                 ):

        layer_dims = [input_features] + layers
        
        self.layers = [
                Layer(layer_dims[i-1], layer_dims[i])
                for i in range(1, len(layer_dims))
                ]
        self.activation = nn.ReLU()
        
        self.classifier = nn.Linear(layer_dims[-1], num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x: TensorType) -> TensorType:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x

    def train(self, train_loader, dev_loader, epochs: int):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        
        for epoch in range(epochs):
            for _, (mb_x, mb_y) in enumerate(train_loader):

                embeddings = self.forward(mb_x)
                y_logits = self.classifier(embeddings)
                y_pred = self.softmax(y_logits)

                loss = criterion(y_pred, mb_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_pos = 0
            total = 0
            for _, (d_mb_x,d_mb_y) in enumerate(dev_loader):
                embeddings = self.forward(d_mb_x)
                y_pred_probabilities = self.softmax(self.classifier(embeddings))
                
                _, dev_predicted = torch.max(y_pred_probabilities, 1)
                
                true_pos += torch.sum(dev_predicted == d_mb_y)
                total += len(d_mb_y)
            dev_acc = true_pos / total
            print("dev acc = %.3f" % (dev_acc*100))


class TweetDataset(Dataset):
    def __init__(self):
        ...

def main():
    ...

if __name__ == '__main__':
    main()



