import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

databody = np.load("databody.npz")
data1 = databody["data1"]
data2 = databody["data2"]
data3 = databody["data3"]
data4 = databody["data4"]
data5 = databody["data5"]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], marker='+', color='b')
ax.scatter(data2[:, 0], data2[:, 1], data3[:, 2], marker='o', color='c')
ax.scatter(data3[:, 0], data3[:, 1], data3[:, 2], marker='*', color='g')
ax.scatter(data4[:, 0], data4[:, 1], data4[:, 2], marker='*', color='r')
ax.scatter(data5[:, 0], data5[:, 1], data5[:, 2], marker='x', color='m')

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_title("Data body")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

data = np.vstack((data1, data2, data3, data4, data5))
labels = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50 + [4]*50)

X = torch.tensor(data, dtype=torch.float32)
Y = torch.tensor(labels, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/{epochs}] Strata: {loss.item():.4f}")

with torch.no_grad():
    output = model(X_test)
    _, predicted = torch.max(output, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f"Presnosť: {accuracy * 100:.2f}%")
