import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

# Load data (replace 'databody.mat' with actual data loading)
data = np.load('databody.npz')

data1, data2, data3, data4, data5 = data['data1'], data['data2'], data['data3'], data['data4'], data['data5']

# Concatenate all data and create labels
X = np.vstack([data1, data2, data3, data4, data5])
y = np.concatenate([
    np.full(len(data1), 0),
    np.full(len(data2), 1),
    np.full(len(data3), 2),
    np.full(len(data4), 3),
    np.full(len(data5), 4)
])

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.25, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define neural network class
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(3, 15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Initialize model, loss, and optimizer
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train model
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(train_loader))
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plot training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid()
plt.show()

# Test model
model.eval()
test_predictions = model(X_test_tensor).detach().numpy()
predicted_labels = np.argmax(test_predictions, axis=1)
actual_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predicted_labels == actual_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Classify new points
new_points = np.array([
    [0.5, 0.2, 1],
    [1, 0.9, 0.8],
    [0.35, 0.1, 0.6],
    [0.8, 0.8, 1],
    [0.1, 0.2, 0.3]
], dtype=np.float32)
new_points_tensor = torch.tensor(new_points)
predictions = model(new_points_tensor).detach().numpy()
predicted_classes = np.argmax(predictions, axis=1) + 1

print('Predicted class labels:', predicted_classes)
