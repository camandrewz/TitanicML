import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Loading the Train Dataset
data = pd.read_csv('Titanic/Data/train.csv')

# Dropping Columns that we are guess are irrelevant
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Giving Sex and Embarked Numerical Values
data["Sex"] = data["Sex"].astype('category').cat.codes
data["Embarked"] = data["Embarked"].astype('category').cat.codes


# Shifting the Survived Column to the Last Space
data.insert(8, 'Survived', data.pop('Survived'))

# Replacing Entries with Missing Elements with The Median
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

print(data)

# Sorting Input and Output Data
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# Split Data Between Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.6, random_state=69)

# Standardize Input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Convert y_train from Pandas Series to Numpy Array
y_train = y_train.to_numpy()

# Ensure PyTorch is Using our GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the Model Parameters
EPOCHS = 500
BATCH_SIZE = len(X_train)
LEARNING_RATE = (1 / EPOCHS)
LAYER_NODES = 6
# Define Custom Dataloaders for PyTorch
# Training Data


class trainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train_data = trainData(torch.Tensor(X_train), torch.Tensor(y_train))

# Testing Data


class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = testData(torch.FloatTensor(X_test))

# Initialize the Dataloaders
train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# Define our Neural Network Architecture


class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 7.
        self.layer_1 = nn.Linear(7, LAYER_NODES)
        self.layer_2 = nn.Linear(LAYER_NODES, LAYER_NODES)
        self.layer_out = nn.Linear(LAYER_NODES, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(LAYER_NODES)
        self.batchnorm2 = nn.BatchNorm1d(LAYER_NODES)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


# Initialize our Optimizer and Choose a Loss Function
model = binaryClassification()
model.to(device)

# Optionally Print our Model Architecture
# print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create a Function to Calculate Model Accuracy


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# Begin Training our Model
model.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

# Testing our Model
y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# Saving our Model for Future Use
torch.save(model, 'Titanic/trained_model')

# Confusion Matrix
confusion_matrix(y_test, y_pred_list)

# Classification Report
print(classification_report(y_test, y_pred_list))
