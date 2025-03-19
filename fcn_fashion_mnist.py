import numpy as np
import torch
from torch import nn
import tqdm

import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

train_dataset = torchvision.datasets.FashionMNIST('/Users/chenab/Downloads/FashionMNIST_Train', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


test_dataset = torchvision.datasets.FashionMNIST('/Users/chenab/Downloads/FashionMNIST_Test', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))



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
                                           
num_train_batches=len(train_batches)
num_val_batches=len(val_batches)
num_test_batches=len(test_batches)


print(num_train_batches)
print(num_val_batches)
print(num_test_batches)

#Sample code to visulaize the first sample in first 16 batches 

batch_num = 0
for train_features, train_labels in train_batches:
    
    if batch_num == 16:
        break    # break here
    
    batch_num = batch_num +1
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")



# Sample code to plot N^2 images from the dataset
def plot_images(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_images(train_dataset.data[:64], 8, "First 64 Training Images" )

#Define your (As Cool As It Gets) Fully Connected Neural Network 
class ACAIGFCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256],dropout_prob=0.2, use_batch_norm=False, init_method=None): 
        super(ACAIGFCN, self).__init__()
        layers = []
        self.linear_layers = []  # Store linear layers for initialization
        in_features = input_dim
        #Define the network layer(s) and activation function(s)
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(in_features, hidden_dim)
            self.linear_layers.append(linear_layer)
            layers.append(linear_layer)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_dim
        output_layer = nn.Linear(in_features, output_dim)
        self.linear_layers.append(output_layer)
        layers.append(output_layer)
        self.net = nn.Sequential(*layers)
        
        if init_method:
            self.init_weights(init_method)
            
    def init_weights(self, init_method):
        if init_method == 'xavier_normal':
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
        #Define how your model propagates the input through the network
        return self.net(input)
        
        

def train_model(model, train_batches, val_batches, epochs = 15, learning_rate = 0.01,
                loss_func = None, optimizer_type = 'sgd'):
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
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

# Function to plot comparison of multiple models
def plot_comparison(results_dict, metric='loss', title_prefix="Comparison of", xlabel="Epoch", ylabel=None):
    if ylabel is None:
        ylabel = "Loss" if metric == 'loss' else "Accuracy"
    
    plt.figure(figsize=(12, 6))
    for label, data in results_dict.items():
        if metric == 'loss':
            plt.plot(range(len(data[0])), data[0], marker='o', label=label)
        else:  # accuracy
            plt.plot(range(len(data[1])), data[1], marker='o', label=label)
    
    plt.title(f"{title_prefix} {ylabel}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Compare different learning rates and optimizers
def run_optimizer_comparison():
    results = {}
    test_accuracies = {}
    
    optimizers = ['sgd', 'adam', 'rmsprop']
    learning_rates = [0.001, 0.01, 0.1]
    
    for opt in optimizers:
        for lr in learning_rates:
            model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], dropout_prob=0.0, use_batch_norm=False)
            
            print(f"\nTraining with {opt.upper()} optimizer, learning rate = {lr}")
            train_loss, val_accuracy = train_model(
                model, train_batches, val_batches, 
                epochs=50, learning_rate=lr, optimizer_type=opt
            )
            
            test_acc = calculate_testaccuracy(model, test_batches)
            
            # Store results
            key = f"{opt} lr={lr}"
            results[key] = (train_loss, val_accuracy)
            test_accuracies[key] = test_acc
    
    return results, test_accuracies

# Compare different regularization techniques
def run_regularization_comparison():
    results = {}
    test_accuracies = {}
    
    # Different dropout rates
    dropout_rates = [0.0, 0.2, 0.5]
    for dropout in dropout_rates:
        model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], 
                         dropout_prob=dropout, use_batch_norm=False)
        
        print(f"\nTraining with dropout = {dropout}")
        train_loss, val_accuracy = train_model(
            model, train_batches, val_batches, 
            epochs=50, learning_rate=0.01, optimizer_type='sgd'
        )
        
        test_acc = calculate_testaccuracy(model, test_batches)
        
        # Store results
        key = f"SGD dropout={dropout}"
        results[key] = (train_loss, val_accuracy)
        test_accuracies[key] = test_acc
    
    # With batch normalization
    model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], 
                     dropout_prob=0.0, use_batch_norm=True)
    
    print("\nTraining with batch normalization")
    train_loss, val_accuracy = train_model(
        model, train_batches, val_batches, 
        epochs=50, learning_rate=0.01, optimizer_type='sgd'
    )
    
    test_acc = calculate_testaccuracy(model, test_batches)
    
    # Store results
    key = "SGD with batch norm"
    results[key] = (train_loss, val_accuracy)
    test_accuracies[key] = test_acc
    
    return results, test_accuracies

# Compare different weight initialization methods
def run_initialization_comparison():
    results = {}
    test_accuracies = {}
    
    # Different initialization methods
    init_methods = [None, 'kaiming_uniform', 'xavier_normal', 'random_normal']
    
    for init in init_methods:
        init_name = init if init else 'default'
        model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], 
                         dropout_prob=0.0, use_batch_norm=False, init_method=init)
        
        print(f"\nTraining with initialization = {init_name}")
        train_loss, val_accuracy = train_model(
            model, train_batches, val_batches, 
            epochs=50, learning_rate=0.01, optimizer_type='sgd'
        )
        
        test_acc = calculate_testaccuracy(model, test_batches)
        
        # Store results
        key = f"{init_name} init"
        results[key] = (train_loss, val_accuracy)
        test_accuracies[key] = test_acc
    
    return results, test_accuracies

# Run final model with best configuration
def run_final_model():
    model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], 
                     dropout_prob=0.2, use_batch_norm=False, init_method='kaiming_uniform')
    
    print("\nTraining final model with best configuration")
    train_loss, val_accuracy = train_model(
        model, train_batches, val_batches, 
        epochs=50, learning_rate=0.001, optimizer_type='adam'
    )
    
    test_acc = calculate_testaccuracy(model, test_batches)
    
    return (train_loss, val_accuracy), test_acc

# Compare models on different datasets (Fashion MNIST vs. MNIST)
def run_dataset_comparison():
    # Load MNIST dataset
    train_dataset_mnist = torchvision.datasets.MNIST('/Users/chenab/Downloads/MNIST_Train', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    test_dataset_mnist = torchvision.datasets.MNIST('/Users/chenab/Downloads/MNIST_Test', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    train_indices_mnist, val_indices_mnist, _, _ = train_test_split(
        range(len(train_dataset_mnist)),
        train_dataset_mnist.targets,
        stratify=train_dataset_mnist.targets,
        test_size=0.1,
    )

    train_split_mnist = Subset(train_dataset_mnist, train_indices_mnist)
    val_split_mnist = Subset(train_dataset_mnist, val_indices_mnist)

    train_batches_mnist = DataLoader(train_split_mnist, batch_size=train_batch_size, shuffle=True)
    val_batches_mnist = DataLoader(val_split_mnist, batch_size=train_batch_size, shuffle=True)
    test_batches_mnist = DataLoader(test_dataset_mnist, batch_size=test_batch_size, shuffle=True)
    
    num_train_batches_mnist = len(train_batches_mnist)
    num_val_batches_mnist = len(val_batches_mnist)
    num_test_batches_mnist = len(test_batches_mnist)
    
    print(num_train_batches_mnist)
    print(num_val_batches_mnist)
    print(num_test_batches_mnist)
    
    # Train on MNIST
    model = ACAIGFCN(input_dim=784, output_dim=10, hidden_dims=[512, 256], 
                     dropout_prob=0.2, use_batch_norm=False, init_method='kaiming_uniform')
    
    print("\nTraining on MNIST dataset")
    train_loss, val_accuracy = train_model(
        model, train_batches_mnist, val_batches_mnist, 
        epochs=50, learning_rate=0.001, optimizer_type='adam'
    )
    
    test_acc = calculate_testaccuracy(model, test_batches_mnist)
    
    return (train_loss, val_accuracy), test_acc

def run_knn_comparison():
    # KNN on Fashion MNIST
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    X_train = []
    y_train = []
    for images, labels in train_batches:
        images = images.view(images.size(0), -1)
        X_train.append(images.numpy())
        y_train.append(labels.numpy())

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test = []
    y_test = []
    for images, labels in test_batches:
        images = images.view(images.size(0), -1)
        X_test.append(images.numpy())
        y_test.append(labels.numpy())

    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN Test accuracy: {:.2f}%".format(accuracy * 100))
    return accuracy

# Plot comparison between multiple models
def plot_multiple_comparison(results_dict, metric='loss'):
    is_loss = metric == 'loss'
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    for label, data in results_dict.items():
        plt.plot(range(len(data[0])), data[0], marker='o', label=label)
    plt.title("Training Loss Comparison", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    for label, data in results_dict.items():
        plt.plot(range(len(data[1])), data[1], marker='o', label=label)
    plt.title("Validation Accuracy Comparison", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Run optimizer comparison (SGD, Adam, RMSProp with different learning rates)
    opt_results, opt_test_acc = run_optimizer_comparison()
    
    # Plot SGD with different learning rates
    sgd_results = {k: v for k, v in opt_results.items() if 'sgd' in k}
    plot_multiple_comparison(sgd_results)
    
    # Plot Adam with different learning rates
    adam_results = {k: v for k, v in opt_results.items() if 'adam' in k}
    plot_multiple_comparison(adam_results)
    
    # Compare Adam and RMSProp
    adam_rmsprop_results = {
        k: v for k, v in opt_results.items() 
        if ('adam' in k or 'rmsprop' in k) and 'lr=0.001' in k or 'lr=0.01' in k
    }
    plot_multiple_comparison(adam_rmsprop_results)
    
    # Run regularization comparison
    reg_results, reg_test_acc = run_regularization_comparison()
    plot_multiple_comparison(reg_results)
    
    # Run initialization methods comparison
    init_results, init_test_acc = run_initialization_comparison()
    plot_multiple_comparison(init_results)
    
    # Run final model with best configuration
    final_results, final_test_acc = run_final_model()
    plot_results(final_results[0], final_results[1], 50)
    
    # Run KNN comparison
    knn_accuracy = run_knn_comparison()
    
    # Run MNIST comparison
    mnist_results, mnist_test_acc = run_dataset_comparison()
    plot_results(mnist_results[0], mnist_results[1], 50)