import numpy as np
import torch
from torch import nn
import tqdm

import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])


train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True, transform=transform)


train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),  
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=0.1,
)

# Generate training and validation subsets based on indices
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)

# set batches sizes
train_batch_size = 512 #Define train batch size
test_batch_size  = 256 #Define test batch size (can be larger than train batch size)

# Define dataloader objects that help to iterate over batches and samples for
# training, validation and testing
train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                                           
num_train_batches = len(train_batches)
num_val_batches = len(val_batches)
num_test_batches = len(test_batches)

print(num_train_batches)
print(num_val_batches)
print(num_test_batches)

#Sample code to visulaize the first sample in first 16 batches 

# batch_num = 0
# for train_features, train_labels in train_batches:
    
#     if batch_num == 16:
#         break    # break here
    
#     batch_num = batch_num +1
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
    
#     img = train_features[0].squeeze()
#     label = train_labels[0]
#     plt.imshow(img, cmap="gray")
#     plt.show()
#     print(f"Label: {label}")



# Sample code to plot N^2 images from the dataset
# def plot_images(XX, N, title):
#     fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
#     for i in range(N):
#       for j in range(N):
#         ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
#         ax[i,j].axis("off")
#     fig.suptitle(title, fontsize=24)

# plot_images(train_dataset.data[:64], 8, "First 64 Training Images" )

    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ACAIGFCN(nn.Module):
    # Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, hidden_dims=[100, 100, 97], dropout_prob=0.0, use_batch_norm=True, init_method = None): 
        super(ACAIGFCN, self).__init__()
        layers = []
        self.linear_layers = []  # Store linear layers for initialization
        in_features = input_dim
        
        # Define the network layer(s) and activation function(s)
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(in_features, hidden_dim)
            self.linear_layers.append(linear_layer)  # Store reference to linear layer
            layers.append(linear_layer)
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_dim
            
        output_layer = nn.Linear(in_features, output_dim)
        self.linear_layers.append(output_layer)  # Add output layer to list
        layers.append(output_layer)
        
        self.net = nn.Sequential(*layers)
        self.init_weights(init_method)
    
    def init_weights(self, init_method):
        if init_method is None:
            return
        elif init_method == 'xavier_normal':
            for layer in self.linear_layers:
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        elif init_method == 'random_normal':
            for layer in self.linear_layers:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        elif init_method == 'kaiming_uniform':
            for layer in self.linear_layers:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")
 
    def forward(self, input):
        # Define how your model propagates the input through the network
        return self.net(input)
    
    
count_parameters(ACAIGFCN(input_dim=784, output_dim=10))


def get_optimizer(model, optimizer_type, learning_rate):
    if optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def train_model(model, train_batches, val_batches, epochs = 15, learning_rate = 0.01,
                loss_func = None, optimizer_type = 'sgd'):
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, learning_rate)
    num_train_batches = len(train_batches)
    train_loss_list = np.zeros((epochs,))
    validation_accuracy_list = np.zeros((epochs,))
    
    for epoch in tqdm.trange(epochs):
        model.train()
        epoch_loss = 0.0
        for train_features, train_labels in train_batches:
            train_features = train_features.reshape(-1, 28*28)
            optimizer.zero_grad()
            outputs = model(train_features)
            loss = loss_func(outputs, train_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / num_train_batches
        train_loss_list[epoch] = avg_loss
        
        #Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_features, val_labels in val_batches:
                val_features = val_features.reshape(-1, 28*28)
                outputs = model(val_features)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == val_labels).sum().item()
                total += val_labels.size(0)
        val_accuracy = correct / total
        validation_accuracy_list[epoch] = val_accuracy
        
        #Print training loss and validation accuracy
        print(f"Epoch {epoch+1}/{epochs}; Train Loss: {avg_loss:.4f}; Validation Accuracy: {val_accuracy*100:.2f}%")
        
    return train_loss_list, validation_accuracy_list

def plot_results(train_loss_list, validation_accuracy_list, epochs):
    # Plot training loss and validation accuracy throughout the training epochs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_list, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), validation_accuracy_list, marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

def calculate_testaccuracy(model, test_batches):
    test_correct = 0
    test_total = 0
    # Telling PyTorch we aren't passing inputs to network for training purpose
    with torch.no_grad():
        
        for test_features, test_labels in test_batches:

            model.eval()
            test_features = test_features.reshape(-1, 28*28)
            outputs = model(test_features)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == test_labels).sum().item()
            test_total += test_labels.size(0)
    test_accuracy = test_correct / test_total
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    return test_accuracy

def calculate_testaccuracy_cnn(model, test_batches):
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for test_features, test_labels in test_batches:
            model.eval()
            test_features = test_features.reshape(-1, 1, 28, 28)
            outputs = model(test_features)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == test_labels).sum().item()
            test_total += test_labels.size(0)
    test_accuracy = test_correct / test_total
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    return test_accuracy

def run_fcn_experiment(optimizer_type, lr, name, model=None):
    if model is None:
        model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
    train_loss, val_acc = train_model(model, train_batches, val_batches, epochs=50, learning_rate=lr, loss_func=None, optimizer_type=optimizer_type)
    test_acc = calculate_testaccuracy(model, test_batches)
    final_acc = val_acc[-1]
    print(f"Final Accuracy for {name} with learning rate {lr}: {final_acc*100:.2f}%")
    return train_loss, val_acc, test_acc, final_acc, model

train_loss_list_sgd_001, validation_accuracy_list_sgd_001, test_accuracy_sgd001, final_accuracy_sgd001, _ = run_fcn_experiment('sgd', 0.001, "SGD")
train_loss_list_sgd_01, validation_accuracy_list_sgd_01, test_accuracy_sgd01, final_accuracy_sgd01, _ = run_fcn_experiment('sgd', 0.01, "SGD")
train_loss_list_sgd_1, validation_accuracy_list_sgd_1, test_accuracy_sgd1, final_accuracy_sgd1, _ = run_fcn_experiment('sgd', 0.1, "SGD")
train_loss_list_adam_001, validation_accuracy_list_adam_001, test_accuracy_adam001, final_accuracy_adam001, _ = run_fcn_experiment('adam', 0.001, "Adam")
model_adam = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_list_adam_01, validation_accuracy_list_adam_01, test_accuracy_adam01, final_accuracy_adam01, model_adam = run_fcn_experiment('adam', 0.01, "Adam", model_adam)
train_loss_list_adam_1, validation_accuracy_list_adam_1, test_accuracy_adam1, final_accuracy_adam1, model_adam = run_fcn_experiment('adam', 0.1, "Adam", model_adam)
test_accuracy_array_leftover = [test_accuracy_adam01, test_accuracy_adam1]
final_accuracy_list_fcn = [final_accuracy_sgd001, final_accuracy_sgd01, final_accuracy_sgd1, final_accuracy_adam001, final_accuracy_adam01, final_accuracy_adam1]
train_loss_list_rmsprop_001, validation_accuracy_list_rmsprop_001, test_accuracy_rmsprop001, final_accuracy_rmsprop001, _ = run_fcn_experiment('rmsprop', 0.001, "RMSprop")
train_loss_list_rmsprop_01, validation_accuracy_list_rmsprop_01, test_accuracy_rmsprop01, final_accuracy_rmsprop01, _ = run_fcn_experiment('rmsprop', 0.01, "RMSprop")
train_loss_list_rmsprop_1, validation_accuracy_list_rmsprop_1, test_accuracy_rmsprop1, final_accuracy_rmsprop1, _ = run_fcn_experiment('rmsprop', 0.1, "RMSprop")
final_accuracy_list_fcn += [final_accuracy_rmsprop001, final_accuracy_rmsprop01, final_accuracy_rmsprop1]

models_array = ["sgd001", "sgd01", "sgd1", "adam001", "adam01", "adam1", "rmsprop001", "rmsprop01", "rmsprop1"]
best_model = models_array[np.argmax(final_accuracy_list_fcn)]
print(f"Best model: {best_model}")

print(test_accuracy_array_leftover)
print(np.array(final_accuracy_list_fcn))

plt.figure(figsize=(16, 7))

sgd_color = '#1f77b4'  # blue
adam_color = '#d62728'  # red
rmsprop_color = '#2ca02c'  # green

plt.subplot(1, 2, 1)

# SGD curves 
plt.plot(range(50), train_loss_list_sgd_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=sgd_color, label='SGD lr=0.001')
plt.plot(range(50), train_loss_list_sgd_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#4c9aff', label='SGD lr=0.01')  # lighter blue


# Adam curves 
plt.plot(range(50), train_loss_list_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=adam_color, label='Adam lr=0.001')
plt.plot(range(50), train_loss_list_adam_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#ff7f7f', label='Adam lr=0.01')  # lighter red


# RMSprop curves
plt.plot(range(50), train_loss_list_rmsprop_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2.5, color=rmsprop_color, label='RMSprop lr=0.001')

plt.title("Training Loss Comparison - All Optimizers", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

# Second subplot for validation accuracy of all optimizers
plt.subplot(1, 2, 2)

# SGD curves 
plt.plot(range(50), validation_accuracy_list_sgd_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=sgd_color, label='SGD lr=0.001')
plt.plot(range(50), validation_accuracy_list_sgd_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#4c9aff', label='SGD lr=0.01')  # lighter blue


# Adam curves 
plt.plot(range(50), validation_accuracy_list_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=adam_color, label='Adam lr=0.001')
plt.plot(range(50), validation_accuracy_list_adam_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#ff7f7f', label='Adam lr=0.01')  # lighter red


# RMSprop curves 
plt.plot(range(50), validation_accuracy_list_rmsprop_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2.5, color=rmsprop_color, label='RMSprop lr=0.001')
plt.plot(range(50), validation_accuracy_list_rmsprop_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2.5, color='#7fd17f', label='RMSprop lr=0.01')  # lighter green


plt.title("Validation Accuracy Comparison - All Optimizers", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

#Comparing Dropout and Batch Normalization on Adam001


#Dropout 0.2, Batch Normalization False
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.2, use_batch_norm=False, init_method='kaiming_uniform') 

train_loss_list_adam_001_dropout_02, validation_accuracy_list_adam_001_dropout_02 = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_adam001_dropout_02 = calculate_testaccuracy(model, test_batches)
final_accuracy_adam_001_dropout_02 = validation_accuracy_list_adam_001_dropout_02[-1]
print(f"Final Accuracy for Adam with learning rate 0.001 and dropout 0.2: {final_accuracy_adam_001_dropout_02*100:.2f}%")

#Dropout 0.5, Batch Normalization False
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.5, use_batch_norm=False, init_method='kaiming_uniform') 

train_loss_list_adam_001_dropout_05, validation_accuracy_list_adam_001_dropout_05 = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_adam001_dropout_05 = calculate_testaccuracy(model, test_batches)
final_accuracy_adam_001_dropout_05 = validation_accuracy_list_adam_001_dropout_05[-1]
print(f"Final Accuracy for Adam with learning rate 0.001 and dropout 0.5: {final_accuracy_adam_001_dropout_05*100:.2f}%")

#Dropout 0.2, Batch Normalization True
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 

train_loss_list_adam_001_dropout_02_bn, validation_accuracy_list_adam_001_dropout_02_bn = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_adam001_dropout_02_bn = calculate_testaccuracy(model, test_batches)
final_accuracy_adam_001_dropout_02_bn = validation_accuracy_list_adam_001_dropout_02_bn[-1]
print(f"Final Accuracy for Adam with learning rate 0.001 and dropout 0.2: {final_accuracy_adam_001_dropout_02*100:.2f}%")

#Dropout 0.5, Batch Normalization True
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.5, use_batch_norm=True, init_method='kaiming_uniform') 

train_loss_list_adam_001_dropout_05_bn, validation_accuracy_list_adam_001_dropout_05_bn = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_adam001_dropout_05_bn = calculate_testaccuracy(model, test_batches)
final_accuracy_adam_001_dropout_05_bn = validation_accuracy_list_adam_001_dropout_05_bn[-1]
print(f"Final Accuracy for Adam with learning rate 0.001 and dropout 0.5: {final_accuracy_adam_001_dropout_05*100:.2f}%")

plt.figure(figsize=(15, 6))

dropout_02_color = 'lightblue'  # blue for dropout 0.2, no BN
dropout_05_color = '#d62728'  # red for dropout 0.5, no BN
dropout_02_bn_color = 'darkblue'  # dark blue for dropout 0.2 with BN
dropout_05_bn_color = '#8b0000'  # dark red for dropout 0.5 with BN

plt.subplot(1, 2, 1)

plt.plot(range(50), train_loss_list_adam_001_dropout_02, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_color, label='Dropout 0.2')
plt.plot(range(50), train_loss_list_adam_001_dropout_05, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_05_color, label='Dropout 0.5')
plt.plot(range(50), train_loss_list_adam_001_dropout_02_bn, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_bn_color, label='Dropout 0.2 + BN')
plt.plot(range(50), train_loss_list_adam_001_dropout_05_bn, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_05_bn_color, label='Dropout 0.5 + BN')

plt.title("Training Loss Comparison - Dropout and Batch Normalization", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

plt.subplot(1, 2, 2)

plt.plot(range(50), validation_accuracy_list_adam_001_dropout_02, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_color, label='Dropout 0.2')
plt.plot(range(50), validation_accuracy_list_adam_001_dropout_05, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_05_color, label='Dropout 0.5')
plt.plot(range(50), validation_accuracy_list_adam_001_dropout_02_bn, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_bn_color, label='Dropout 0.2 + BN')
plt.plot(range(50), validation_accuracy_list_adam_001_dropout_05_bn, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_05_bn_color, label='Dropout 0.5 + BN')

plt.title("Validation Accuracy Comparison - Dropout and Batch Normalization", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

plt.tight_layout()
plt.show()

#50K
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[50,95,50], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 

num_weights = count_parameters(model)
print(f"Number of weights in the model: {num_weights}")

#50K
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[50,95,50], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_50k, validation_accuracy_list_50k = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_50k = calculate_testaccuracy(model, test_batches)
print(f"Test Accuracy for Adam with learning rate 0.001 and dropout 0.2 and batch normalization: {test_accuracy_50k*100:.2f}%")

#50K learning rate 0.01
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[50,95,50], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_50k_lr01, validation_accuracy_list_50k_lr01 = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.01, loss_func=None, optimizer_type='adam')
test_accuracy_50k_lr01 = calculate_testaccuracy(model, test_batches)

#200K model
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[200,90,200], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 

num_weights = count_parameters(model)
print(f"Number of weights in the model: {num_weights}")

training_loss_list_200k, validation_accuracy_list_200k = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_200k = calculate_testaccuracy(model, test_batches)

#200K learning rate 0.1
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[200,90,200], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_50k_lr1, validation_accuracy_list_50k_lr1 = train_model(model, train_batches, val_batches, epochs=50, learning_rate=0.1, loss_func=None, optimizer_type='adam')
test_accuracy_50k_lr1 = calculate_testaccuracy(model, test_batches)

#50K 100 epochs
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[50,95,50], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_50k_100epochs, validation_accuracy_list_50k_100epochs = train_model(model, train_batches, val_batches, epochs=100, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_50k_100epochs = calculate_testaccuracy(model, test_batches)
print(f"Test Accuracy for Adam with learning rate 0.001 and dropout 0.2 and batch normalization: {test_accuracy_50k_100epochs*100:.2f}%")

#100K 100 epochs
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[100,100,97], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_100k_100epochs, validation_accuracy_list_100k_100epochs = train_model(model, train_batches, val_batches, epochs=100, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_100k_100epochs = calculate_testaccuracy(model, test_batches)
print(f"Test Accuracy for Adam with learning rate 0.001 and dropout 0.2 and batch normalization: {test_accuracy_100k_100epochs*100:.2f}%")

#200K 100 epochs
model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[200,90,200], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform') 
training_loss_list_200k_100epochs, validation_accuracy_list_200k_100epochs = train_model(model, train_batches, val_batches, epochs=100, learning_rate=0.001, loss_func=None, optimizer_type='adam')
test_accuracy_200k_100epochs = calculate_testaccuracy(model, test_batches)
print(f"Test Accuracy for Adam with learning rate 0.001 and dropout 0.2 and batch normalization: {test_accuracy_200k_100epochs*100:.2f}%")

plt.figure(figsize=(16, 7))


model_50k_color = 'darkblue'  
model_100k_color = 'red'  
model_200k_color = 'green'
# First subplot for training loss
plt.subplot(1, 2, 1)

# Plot training loss for different model sizes 100 epochs   
plt.plot(range(100), training_loss_list_50k_100epochs, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_50k_color, label='50K Parameters Model')
plt.plot(range(100), training_loss_list_100k_100epochs, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_100k_color, label='100K Parameters Model')
plt.plot(range(100), training_loss_list_200k_100epochs, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_200k_color, label='200K Parameters Model')

plt.title("Training Loss Comparison - Model Size", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

# Second subplot for validation accuracy
plt.subplot(1, 2, 2)

# Plot validation accuracy for different model sizes
plt.plot(range(100), validation_accuracy_list_50k_100epochs, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_50k_color, label='50K Parameters Model')
plt.plot(range(100), validation_accuracy_list_100k_100epochs, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_100k_color, label='100K Parameters Model')
plt.plot(range(100), validation_accuracy_list_200k_100epochs, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_200k_color, label='200K Parameters Model')
plt.title("Validation Accuracy Comparison - Model Size", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()