import sklearn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt

# Loading and processing the dataset
df = pd.read_csv("chronickidneydiseases.csv")
X = df.iloc[:, :-1].values  # All columns except the last one (features)
y = df['RecommendedVisitsPerMonth'].values  # The target variable

# If target variable (y) is categorical, encode it
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize the feature set
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset instances
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Federated Learning Client Class
class Client:
    def __init__(self, client_id, dataset, device):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def train(self, global_model, epochs, lr):
        model = copy.deepcopy(global_model)
        model.to(self.device)  # Ensure the model is on the correct device
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
            print(f'Client {self.client_id}, Epoch {epoch + 1}, Loss: {running_loss / len(train_loader.dataset)}')

        return model.state_dict()

# Federated Averaging Function
def federated_average(client_weights):
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
    return avg_weights

# Training setup
num_clients = 5
epochs = 5
lr = 0.01
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split dataset among clients
base_size = len(train_dataset) // num_clients
remainder = len(train_dataset) % num_clients
split_sizes = [base_size + 1 if i < remainder else base_size for i in range(num_clients)]
client_datasets = torch.utils.data.random_split(train_dataset, split_sizes)

# Initialize clients
clients = [Client(client_id=i, dataset=client_datasets[i], device=device) for i in range(num_clients)]

# Initialize global model
input_size = X_train.shape[1]
output_size = len(set(y))
global_model = SimpleNN(input_size, output_size).to(device)

# Initialize tracking variables
client_losses = {i: [] for i in range(num_clients)}
validation_accuracies = []

# Federated Learning Training Loop
for round_num in range(epochs):
    print(f"--- Round {round_num + 1} ---")

    # Each client trains on their local data
    client_weights = []

    for idx, client in enumerate(clients):
        # Get model update and track loss
        model = copy.deepcopy(global_model)
        model.to(device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        train_loader = DataLoader(client.dataset, batch_size=batch_size, shuffle=True)

        epoch_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Store average loss for this client
        avg_loss = epoch_loss / len(train_loader)
        client_losses[idx].append(avg_loss)

        # Add model weights to collection
        client_weights.append(model.state_dict())
        print(f'Client {idx}, Round {round_num + 1}, Loss: {avg_loss:.4f}')

    # Federated averaging to update the global model
    global_weights = federated_average(client_weights)
    global_model.load_state_dict(global_weights)

    # Evaluate global model
    global_model.eval()
    correct = 0
    total = 0
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = global_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    validation_accuracies.append(accuracy)
    print(f"Validation Accuracy after Round {round_num + 1}: {accuracy:.2f}%")
torch.save(global_model.state_dict(), "federated_model.pth")

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Training Progress Over Time
plt.subplot(1, 3, 1)
epochs_range = range(1, epochs + 1)
for client_id, losses in client_losses.items():
    plt.plot(epochs_range, losses, marker='o', label=f'Client {client_id}')
plt.xlabel('Rounds')
plt.ylabel('Training Loss')
plt.title('Client Training Loss Over Time')
plt.legend()
plt.grid(True)

# Plot 2: Model Performance
plt.subplot(1, 3, 2)
plt.plot(epochs_range, validation_accuracies, 'b-', marker='s')
plt.fill_between(epochs_range,
                 [max(0, acc - 5) for acc in validation_accuracies],
                 [min(100, acc + 5) for acc in validation_accuracies],
                 alpha=0.2)
plt.xlabel('Rounds')
plt.ylabel('Validation Accuracy (%)')
plt.title('Global Model Accuracy')
plt.grid(True)

# Plot 3: Parameter Distribution
plt.subplot(1, 3, 3)
weights = []
for param in global_model.parameters():
    weights.extend(param.cpu().detach().numpy().flatten())
plt.hist(weights, bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Weight Values')
plt.ylabel('Frequency')
plt.title('Model Weight Distribution')
plt.grid(True)

plt.tight_layout()
plt.show()