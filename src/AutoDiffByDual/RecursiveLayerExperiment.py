#pip install torch torchvision matplotlib tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import math

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 10
n_layers = 3  # Specify the number of layers as a hyperparameter
hidden_size = 128  # Size of the hidden layers

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Neural Network with Dynamic Number of Layers
class Baseline(nn.Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 n_hid_layers, 
                 hidden_size):
        super(Baseline, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_hid_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
      x = x.view(x.size(0), -1) 
      return self.model(x)

class ContinuousIdentity(nn.Module):
  def __init__(self, T):
    super(ContinuousIdentity, self).__init__()
    self.T = T
  def forward(self, i, j):
    return 1.0 / math.e ** (self.T * (i-j)**2)

class RecursiveLayer(nn.Module):
    def __init__(self, n_layers, io_size, T=10, k=2):
        super(RecursiveLayer, self).__init__()
        #self.r = torch.Tensor.float(1.0)
        self.r = nn.Parameter(torch.tensor(1.0))
        layers = [nn.Linear(io_size, io_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(io_size, io_size), nn.ReLU()]
        self.model = nn.Sequential(*layers)
        self.I = ContinuousIdentity(T)
        self.k = k

    def forward(self, x):
      y=0
      for i in range(1, math.ceil(self.r.item())+self.k):
          x = self.model(x)
          y = y + self.I(i, self.r) * x
      return y

class RecursiveNetwork(nn.Module):
  def __init__(self, 
               input_size, 
               output_size,
               nb_per_rec, 
               nb_rec, 
               hidden_size, 
               T=10, 
               k=2):
        super(RecursiveNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(nb_rec):
            layers += [RecursiveLayer(nb_per_rec, hidden_size, T, k)]
        layers += [nn.Linear(hidden_size, output_size)]
        self.model = nn.Sequential(*layers)
  def forward(self, x):
      x = x.view(x.size(0), -1) 
      return self.model(x)



def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Move to device
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = scores.max(1)
            total_correct += predictions.eq(targets).sum().item()
            total_samples += predictions.size(0)
        accuracy = float(total_correct) / total_samples
        print(f"Epoch [{epoch+1}/{epochs}]: Training Accuracy: {accuracy*100:.2f}%")

        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Hyperparameters
hidden_size = 16  # Size of the hidden layers
nb_per_rec = 2
nb_rec = 5
n_layers = nb_per_rec * nb_rec 
Temperature = 1
k = 10

baseline = Baseline(input_size=28*28, 
                    output_size=10, 
                    n_hid_layers=10, 
                    hidden_size=hidden_size,
                    ).to(device)

recursive = RecursiveNetwork(
   input_size=28*28, 
   output_size=10, 
   nb_per_rec= nb_per_rec,
   nb_rec = nb_rec,
   hidden_size=hidden_size,
   T=Temperature,
   k=k
   ).to(device)

# Learning hyperparameters
learning_rate = 0.001
epochs = 5


def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_correct = 0
        total_samples = 0
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Move to device
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predictions = scores.max(1)
            total_correct += predictions.eq(targets).sum().item()
            total_samples += predictions.size(0)
        accuracy = float(total_correct) / total_samples
        print(f"Epoch [{epoch+1}/{epochs}]: Training Accuracy: {accuracy*100:.2f}%")

train_model(baseline, train_loader, epochs)
train_model(recursive, train_loader, epochs)

# Test the model
# (Add code here to evaluate the model on the test dataset)

# Save the model
# torch.save(model.state_dict(), 'flexible_nn.pth')