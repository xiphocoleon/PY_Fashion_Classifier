import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets, transforms 
from torchvision.transforms import ToTensor, Lambda 
import matplotlib.pyplot as plt 

# Device setup
# Set all tensors to the first CUDA device
device = torch.device("cuda:0")
torch.set_default_device(device) 
print(f"The device used is {device}")

### START CLASSES ###
# Custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path) # Converts image into a tensor
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
       )
      
    def forward(self, x):
        x = x.to(device)
        #print(f"x is on this device: {x.get_device()}")
        x = self.flatten(x)
        #print(f"x in the Neural Network object is on this device: {x.get_device()}")
        logits = self.linear_relu_stack(x)
        return logits
### END CLASSES ###

### START FUNCTIONS - TEST AND TRAIN LOOPS ###
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode, important for batch normaliz'n and dropout layers
    # Unnecessary in this situaion but added for best practice
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if batch % 100 == 0:
        loss, current = loss.item(), batch * batch_size + len(X)
        print(f"loss: {loss:>7f} [{current:>5}|{size:>5}]")
        
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode -- important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practice
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # Also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=true

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
### END FUNCTIONS ###

### START MAIN SEQUENCE ###
# Set model
model = NeuralNetwork().to(device)
print(model)

# Assign Training and Test Data and use Dataloader to load it
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n________________________________________")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    
print("DONE")


"""

Inside the training loop, optimization happens in three steps:

Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.

Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.

Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.

"""


"""
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient functin for loss = {loss.grad_fn}")
loss.backward()
print(w.grad)
print(b.grad)
"""

### END VARIABLES ###

### NN CALL AND TESTS ###
"""
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device = device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

input_image = torch.rand(3, 28, 28)
print(input_image.size())
"""

# Tests with different layers
"""
# Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())   

# ReLU
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequential
seq_modules = nn.Sequential(
    flatten,
    layer1, 
    nn.ReLU(),
    nn.Linear(20,10)
    )
input_image = torch.rand(3, 28, 28) 
logits = seq_modules(input_image)

# Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
"""


# Dataset prepared for training with features as normalized tensores
# and labels as one-hot encoded tensors, using tranformations with ToTensor
# and Lambda
"""
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,
        torch.tensor(y), value=1))
    )


labels_map = {
    0: "T-Shirt", 
    1: "Trouser",
    2: "Pullover", 
    3: "Dress", 
    4: "Coat", 
    5: "Sandal", 
    6: "Shirt", 
    7: "Sneaker", 
    8: "Bag", 
    9: "Ankle Boot", 
    }

"""
"""
Visualize with a grid of labeled images
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1, )).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
"""
"""


# Display single image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

"""

"""

x = torch.ones(5) # test input tensor
y = torch.zeros(3)  # test output tensor
w = torch.randn(5, 3, requires_grad=True)  # weight param
b = torch.randn(3, requires_grad=True) # offset param
z = torch.matmul(x, w) + b
#loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

learning_rate = 1e-3
batch_size = 64
epochs = 5
"""