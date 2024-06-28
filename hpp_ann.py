#%% Importing the libraries

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import torch.nn as nn
import scipy.sparse
import torch.optim as optim

#%% Loading the Train set

df_train = pd.read_csv('train.csv')
X_train = df_train.drop(columns=['SalePrice'])
y_train = df_train['SalePrice']

#%% Loading the test set

df_test = pd.read_csv('test.csv')
X_test = df_test.copy()

#%% Identifying the numerical and categorical columns

numerical_cols = X_train.select_dtypes(
    include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

#%% Imputing the missing data

num_imputer = SimpleImputer(strategy='mean')
X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

#%% Feature Scaling

scaler = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)], 
    remainder='passthrough')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Initalizing the parameters

input_dims = X_train.shape[1]
hidden_dims1 = 64
hidden_dims2 = 64
output_dims = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Creating and initializing the ANN

class ANN(nn.Module):
    def __init__(self, input_dims, hidden_dims1, hidden_dims2, output_dims):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims1)
        self.fc2 = nn.Linear(hidden_dims1, hidden_dims2)
        self.fc3 = nn.Linear(hidden_dims2, output_dims)
        
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        
        return x

model = ANN(input_dims, hidden_dims1, hidden_dims2, output_dims).to(device)


#%% Converting the data into tensors and moving it CUDA

if isinstance(X_train, scipy.sparse.csr_matrix):
    X_train = X_train.toarray()

if isinstance(X_test, scipy.sparse.csr_matrix):
    X_test = X_test.toarray()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

#%% Initializing the Loss functiona and Optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#%% Training Loop

num_epochs = 10000

for epoch in range(num_epochs):
    model.train()
    
    # Running the model
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#%% Evaluating the model and making predictions
model.eval()
with torch.no_grad():
    train_predictions = model(X_train_tensor)
    train_loss = criterion(train_predictions, y_train_tensor)
    print(f'Training Loss: {train_loss.item():.4f}')
    
    # Predict on test data
    test_predictions = model(X_test_tensor)

test_predictions = test_predictions.cpu().numpy()


