import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("Concrete_Data.csv")

# Separating data - features from targets 
x_data = data.iloc[:, :-1].values
y_data = data.iloc[:,-1].values.reshape(-1,1)

# standardizing the features 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_data)

# converting from dataframes to tensors
x_tens = torch.tensor(x_scaled, dtype=torch.float32)
y_tens = torch.tensor(y_data, dtype=torch.float32)

# set seed and shuffle our data to prevent biased output
torch.manual_seed(42)
num_samples = x_tens.shape[0]
indices = torch.randperm(num_samples)

# splitting the data into training and testing datasets 80/20 split
split_idx = int(0.8* num_samples)
train = indices[:split_idx]
test = indices[split_idx:]

# getting x train and y train and x test and y test
x_train = x_tens[train]
y_train = y_tens[train]
x_test = x_tens[test]
y_test = y_tens[test]

# ----------Data Preparation step is done --------------#

class ConcreteMLP(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(ConcreteMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.relu = nn.ReLU()                          # Activation
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer
    
    def forward(self, x):
        x = self.fc1(x)   # Apply first linear layer
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)   # Apply output layer
        return x

# Custom Model has been defined 

model = ConcreteMLP()

#  setup loss and optimizer- carries the learning rate and model parameters as an argument
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------Now Writing the training loop----------#
num_epochs = 300

training_loss = []

for epoch in range(num_epochs):
    # pass it through model
    y_pred = model(x_train)

    # compute loss
    loss = loss_fn(y_pred, y_train)

    # backward loss
    loss.backward()

    # step optimizer
    optimizer.step()

    # zero gradient
    optimizer.zero_grad()

    training_loss.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate model performance

with torch.no_grad():
    # Predict on train set
    train_preds = model(x_train)
    train_loss = loss_fn(train_preds, y_train)

    # Predict on test set
    test_preds = model(x_test)
    test_loss = loss_fn(test_preds, y_test)

print(f"Final Training Loss (MSE): {train_loss.item():.4f}")
print(f"Final Testing Loss (MSE): {test_loss.item():.4f}")

# Calculate RMSE
train_rmse = torch.sqrt(train_loss)
test_rmse = torch.sqrt(test_loss)

print(f"Training RMSE: {train_rmse.item():.2f} MPa")
print(f"Testing RMSE: {test_rmse.item():.2f} MPa")

# Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(num_epochs), training_loss, linestyle='-', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()