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



#######################
##########CNN##########
#######################


class ACAIGCNN(nn.Module):
    def __init__(self, input_dim, output_dim, conv_channels=[16, 32, 64], fc_sizes=[128], dropout_prob=0.2, use_batch_norm=False, init_method = None):
        super(ACAIGCNN, self).__init__()
        self.conv_layers = nn.Sequential()
        in_channels = 1
        
        # Convolutional Layers
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout_prob))
            in_channels = out_channels
        
        self.flattened_size = conv_channels[-1] * 3 * 3
        
        # Fully Connected Layers
        self.fc_layers = nn.Sequential()
        in_features = self.flattened_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(in_features, fc_size))
            if use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(fc_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout_prob))
            in_features = fc_size
        self.fc_layers.append(nn.Linear(in_features, output_dim))
        
        # Initialize weights
        self.init_weights(init_method)
    
    def init_weights(self, init_method):
        if init_method is None:
            return
        if init_method == 'kaiming_uniform':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if init_method == 'random_normal':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if init_method == 'xavier_normal':
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
    
    
model_cnn_50k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[8, 16, 32], fc_sizes=[128], dropout_prob=0.2, use_batch_norm=True)
print(f"CNN 50K Parameters: {count_parameters(model_cnn_50k)}")

model_cnn_20k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[4, 8, 16], fc_sizes=[116], dropout_prob=0.2, use_batch_norm=True)
print(f"CNN 20K Parameters: {count_parameters(model_cnn_20k)}")

model_cnn_10k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[2, 4, 8], fc_sizes=[108], dropout_prob=0.2, use_batch_norm=True)
print(f"CNN 10K Parameters: {count_parameters(model_cnn_10k)}")

def train_cnn_model(model, train_batches, val_batches, epochs = 15, learning_rate = 0.01,
                loss_func = None, optimizer_type = 'sgd', init_method = None):
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
            train_features = train_features.reshape(-1, 1, 28, 28)
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
                val_features = val_features.reshape(-1,1,28,28)
                outputs = model(val_features)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == val_labels).sum().item()
                total += val_labels.size(0)
        val_accuracy = correct / total
        validation_accuracy_list[epoch] = val_accuracy
        
        #Print training loss and validation accuracy
        print(f"Epoch {epoch+1}/{epochs}; Train Loss: {avg_loss:.4f}; Validation Accuracy: {val_accuracy*100:.2f}%")
        
    return train_loss_list, validation_accuracy_list

def run_cnn_experiment(optimizer_type, lr, name, model=None):
    if model is None:
        model = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
    train_loss, val_acc = train_cnn_model(model, train_batches, val_batches, epochs=50, learning_rate=lr, loss_func=None, optimizer_type=optimizer_type, init_method='kaiming_uniform')
    test_acc = calculate_testaccuracy_cnn(model, test_batches)
    final_acc = val_acc[-1]
    print(f"Final Accuracy for {name} with learning rate {lr}: {final_acc*100:.2f}%")
    return train_loss, val_acc, test_acc, final_acc, model

import time

time_array = []
test_accuracy_array = []
final_validation_accuracy_array = []

#SGD with learning rate 0.001
model_cnn_100k_sgd_001 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_sgd_001, val_acc_cnn100k_sgd_001, test_accuracy_cnn100k_sgd_001, final_val_acc_cnn100k_sgd_001, _ = run_cnn_experiment('sgd', 0.001, "SGD")
#SGD with learning rate 0.01
model_cnn_100k_sgd_01 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_sgd_01, val_acc_cnn100k_sgd_01, test_accuracy_cnn100k_sgd_01, final_val_acc_cnn100k_sgd_01, _ = run_cnn_experiment('sgd', 0.01, "SGD")
start_time = time.time()
#ADAM with learning rate 0.001
model_cnn_100k_adam_001 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_adam_001, val_acc_cnn100k_adam_001, test_accuracy_cnn100k_adam_001, final_val_acc_cnn100k_adam_001, _ = run_cnn_experiment('adam', 0.001, "Adam")
end_time = time.time()
time_adam_001 = end_time - start_time
time_array.append(time_adam_001)
#ADAM with learning rate 0.01
model_cnn_100k_adam_01 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_adam_01, val_acc_cnn100k_adam_01, test_accuracy_cnn100k_adam_01, final_val_acc_cnn100k_adam_01, _ = run_cnn_experiment('adam', 0.01, "Adam")
#RMSPROP with learning rate 0.001
model_cnn_100k_rmsprop_001 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_rmsprop_001, val_acc_cnn100k_rmsprop_001, test_accuracy_cnn100k_rmsprop_001, final_val_acc_cnn100k_rmsprop_001, _ = run_cnn_experiment('rmsprop', 0.001, "RMSprop")
#RMSPROP with learning rate 0.01
model_cnn_100k_rmsprop_01 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.0, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_rmsprop_01, val_acc_cnn100k_rmsprop_01, test_accuracy_cnn100k_rmsprop_01, final_val_acc_cnn100k_rmsprop_01, _ = run_cnn_experiment('rmsprop', 0.01, "RMSprop")
#ADAM with learning rate 0.001, DROPOUT 0.2, BATCH NORMALIZATION = TRUE
model_cnn_100k_adam_001_dropout_02 = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform')
train_loss_cnn100k_adam_001_dropout_02, val_acc_cnn100k_adam_001_dropout_02, test_accuracy_cnn100k_adam_001_dropout_02, final_val_acc_cnn100k_adam_001_dropout_02, _ = run_cnn_experiment('adam', 0.001, "Adam")
#ADAM with learning rate 0.001, DROPOUT 0.2, BATCH NORMALIZATION = FALSE
model_cnn_100k_adam_001_dropout_02_bn_false = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[16,32,64], fc_sizes=[128], dropout_prob=0.2, use_batch_norm=False, init_method='kaiming_uniform')
train_loss_cnn100k_adam_001_dropout_02_bn_false, val_acc_cnn100k_adam_001_dropout_02_bn_false, test_accuracy_cnn100k_adam_001_dropout_02_bn_false, final_val_acc_cnn100k_adam_001_dropout_02_bn_false, _ = run_cnn_experiment('adam', 0.001, "Adam")

test_accuracy_other_models = []
final_validation_accuracy_other_models = []
time_other_models = []

#CNN 50K
start_time = time.time()
model_cnn_50k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[8,16,32], fc_sizes=[128], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform')
train_loss_cnn50k, val_acc_cnn50k, test_accuracy_cnn50k, final_val_acc_cnn50k, _ = run_cnn_experiment('adam', 0.001, "Adam")
end_time = time.time()
time_cnn50k = end_time - start_time
time_other_models.append(time_cnn50k)
test_accuracy_other_models.append(test_accuracy_cnn50k)
final_validation_accuracy_other_models.append(final_val_acc_cnn50k)

#CNN 
start_time = time.time()
model_cnn_20k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[4,8,16], fc_sizes=[116], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform')
train_loss_cnn20k, val_acc_cnn20k, test_accuracy_cnn20k, final_val_acc_cnn20k, _ = run_cnn_experiment('adam', 0.001, "Adam")
end_time = time.time()
time_cnn20k = end_time - start_time
time_other_models.append(time_cnn20k)
test_accuracy_other_models.append(test_accuracy_cnn20k)
final_validation_accuracy_other_models.append(final_val_acc_cnn20k)

#CNN 10K
start_time = time.time()
model_cnn_10k = ACAIGCNN(input_dim=784, output_dim=10, conv_channels=[2,4,8], fc_sizes=[108], dropout_prob=0.2, use_batch_norm=True, init_method='kaiming_uniform')
train_loss_cnn10k, val_acc_cnn10k, test_accuracy_cnn10k, final_val_acc_cnn10k, _ = run_cnn_experiment('adam', 0.001, "Adam")
end_time = time.time()
time_cnn10k = end_time - start_time
time_other_models.append(time_cnn10k)
test_accuracy_other_models.append(test_accuracy_cnn10k)
final_validation_accuracy_other_models.append(final_val_acc_cnn10k)

print(test_accuracy_array)
print(test_accuracy_other_models)
print(time_array)
print(time_other_models)
print(final_validation_accuracy_array)
print(final_validation_accuracy_other_models)

plt.figure(figsize=(16, 7))

sgd_color = '#1f77b4'  # blue
adam_color = '#d62728'  # red
rmsprop_color = '#2ca02c'  # green

plt.subplot(1, 2, 1)

# SGD curves 
plt.plot(range(50), train_loss_cnn100k_sgd_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=sgd_color, label='SGD lr=0.001')
plt.plot(range(50), train_loss_cnn100k_sgd_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#4c9aff', label='SGD lr=0.01')  # lighter blue


# Adam curves
plt.plot(range(50), train_loss_cnn100k_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=adam_color, label='Adam lr=0.001')
plt.plot(range(50), train_loss_cnn100k_adam_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#ff7f7f', label='Adam lr=0.01')  # lighter red


# RMSprop curves
plt.plot(range(50), train_loss_cnn100k_rmsprop_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2.5, color=rmsprop_color, label='RMSprop lr=0.001')
#plt.plot(range(50), train_loss_cnn100k_rmsprop_01, marker='s', markersize=5, markevery=3, linestyle='-', 
#         linewidth=2.5, color='#7fd17f', label='RMSprop lr=0.01')  # lighter green



plt.title("Training Loss Comparison - All Optimizers", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

plt.subplot(1, 2, 2)

# SGD curves 
plt.plot(range(50), val_acc_cnn100k_sgd_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=sgd_color, label='SGD lr=0.001')
plt.plot(range(50), val_acc_cnn100k_sgd_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#4c9aff', label='SGD lr=0.01')  # lighter blue


# Adam curves 
plt.plot(range(50), val_acc_cnn100k_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=adam_color, label='Adam lr=0.001')
plt.plot(range(50), val_acc_cnn100k_adam_01, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#ff7f7f', label='Adam lr=0.01')  # lighter red


# RMSprop curves 
plt.plot(range(50), val_acc_cnn100k_rmsprop_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2.5, color=rmsprop_color, label='RMSprop lr=0.001')
plt.plot(range(50), val_acc_cnn100k_rmsprop_01, marker='s', markersize=5, markevery=3, linestyle='-',  linewidth=2.5, color='#7fd17f', label='RMSprop lr=0.01')  # lighter green


plt.title("Validation Accuracy Comparison - All Optimizers", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))

dropout_02_color = 'lightblue'  # blue for dropout 0.2, no BN
dropout_05_color = '#d62728'  # red for dropout 0.5, no BN
dropout_02_bn_color = 'darkblue'  # dark blue for dropout 0.2 with BN
dropout_05_bn_color = '#8b0000'  # dark red for dropout 0.5 with BN

plt.subplot(1, 2, 1)

plt.plot(range(50), train_loss_cnn100k_adam_001_dropout_02, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_color, label='Dropout 0.2 + BN')
plt.plot(range(50), train_loss_cnn100k_adam_001_dropout_02_bn_false, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_bn_color, label='Dropout 0.2')
plt.plot(range(50), train_loss_cnn100k_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#8b0000', label='Adam lr=0.001, No Dropout')

plt.title("Training Loss Comparison - Dropout and Batch Normalization", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

plt.subplot(1, 2, 2)

plt.plot(range(50), val_acc_cnn100k_adam_001_dropout_02, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_color, label='Dropout 0.2 + BN')
plt.plot(range(50), val_acc_cnn100k_adam_001_dropout_02_bn_false, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=dropout_02_bn_color, label='Dropout 0.2')
plt.plot(range(50), val_acc_cnn100k_adam_001, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color='#8b0000', label='Adam lr=0.001, No Dropout')

plt.title("Validation Accuracy Comparison - Dropout and Batch Normalization", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 7))

model_10k_color = 'purple'
model_20k_color = 'darkgreen'
model_50k_color = 'darkblue'  
model_100k_color = 'red'


plt.subplot(1, 2, 1)

plt.plot(range(50), train_loss_cnn10k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_10k_color, label='10K Parameters Model')
plt.plot(range(50), train_loss_cnn20k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_20k_color, label='20K Parameters Model')
plt.plot(range(50), train_loss_cnn50k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_50k_color, label='50K Parameters Model')
plt.plot(range(50), train_loss_cnn100k_adam_001_dropout_02, marker='s', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_100k_color, label='100K Parameters Model')


plt.title("Training Loss Comparison for CNN - Model Size", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='upper right')

plt.subplot(1, 2, 2)

plt.plot(range(50), val_acc_cnn10k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_10k_color, label='10K Parameters Model')
plt.plot(range(50), val_acc_cnn20k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_20k_color, label='100K Parameters Model')
plt.plot(range(50), val_acc_cnn50k, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_50k_color, label='50K Parameters Model')
plt.plot(range(50), val_acc_cnn100k_adam_001_dropout_02, marker='o', markersize=5, markevery=3, linestyle='-', 
         linewidth=2, color=model_100k_color, label='100K Parameters Model')
plt.title("Validation Accuracy Comparison for CNN - Model Size", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, framealpha=0.9, loc='lower right')

plt.tight_layout()
plt.show()