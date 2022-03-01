import numpy as np

import torch
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullyConnectedAutoencoder(nn.Module):
  

    def __init__(self, layer_widths, activation_function, seed=1):
        super(FullyConnectedAutoencoder, self).__init__()
        
        # set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        self.data_size = layer_widths[0]
        self.code_size = layer_widths[-1]
        
        self.encode_widths = layer_widths
        self.decode_widths = layer_widths[::-1]
        
        self.activation_function = activation_function

        self.encode_linear_layers = nn.ModuleList([nn.Linear(w[0], w[1]) for w in zip(self.encode_widths[:-1], self.encode_widths[1:])])
        self.decode_linear_layers = nn.ModuleList([nn.Linear(w[0], w[1]) for w in zip(self.decode_widths[:-1], self.decode_widths[1:])])
        
        self.encode_activation_layers = nn.ModuleList([self.activation_function for a in range(len(self.encode_widths) - 1)])
        self.decode_activation_layers = nn.ModuleList([self.activation_function for a in range(len(self.decode_widths) - 2)])

        
    def encode(self, x):

        # State s is input data x
        s = x

        # Iterate all layers
        for linear, activation in zip(self.encode_linear_layers, self.encode_activation_layers):
            s = activation(linear(s))
        
        # Code h
        h = s
        
        return h
    
    
    def decode(self, h):
        
        # State s is code data h
        s = h
        
        # Iterate all layers
        for linear, activation in zip(self.decode_linear_layers[:-1], self.decode_activation_layers):
            s = activation(linear(s))
        
        # Reconstruction x_r
        x_r = self.decode_linear_layers[-1](s)
        
        return x_r
    
    
    def forward(self, x):
        
        # Get code h
        h = self.encode(x)
        
        # Get reconstruction x_r
        x_r = self.decode(h)
    
        return x_r

    
    def train_model(self, loss, lr, batch_size, num_epochs, train_dataset, test_dataset):

        # Loss and optimizer
        self.loss = loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Create PyTorch DataLoader for train and test data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = len(list(test_dataset)), shuffle = False, drop_last = True)
        
        # set our model in the training mode
        self.train()
        
        print("Epoch MSELoss(train) MSELoss(test)")
        for epoch in range(num_epochs):

            epoch_loss_train = 0
            # Compute loss for train set
            # Update weights
            for i, batch_sample in enumerate(train_loader):
        
                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                prediction = self(x_batch)

                l = self.loss(prediction, y_batch)
                epoch_loss_train += l.item()

                self.zero_grad()
                l.backward()
                self.optimizer.step()
            epoch_loss_train /= len(train_loader)
            
            epoch_loss_test = 0
            # Compute loss for test set
            for i, batch_sample in enumerate(test_loader):
        
                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
        
                prediction = self(x_batch)
        
                l = self.loss(prediction, y_batch)
                epoch_loss_test += l.item()
            epoch_loss_test /= len(test_loader)
            
            if epoch % 100 == 0:
                print(str(epoch) + " " + "{:.6f}".format(epoch_loss_train) + " " + "{:.6f}".format(epoch_loss_test))


    def save_model(self, path):

        torch.save(self.state_dict(), path + ".pt")


    def load_model(self, path):        
        
        self.load_state_dict(torch.load(path + ".pt"))
        self.eval()
        return self