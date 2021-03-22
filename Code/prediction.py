import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Loading the Test Dataset
data = pd.read_csv('Titanic/Data/test.csv')
submission = data

# Dropping Columns that we are guess are irrelevant
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
submission = submission.drop(['Pclass', 'Name', 'Sex', 'Age',
                              'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

# Giving Sex and Embarked Numerical Values
data["Sex"] = data["Sex"].astype('category').cat.codes
data["Embarked"] = data["Embarked"].astype('category').cat.codes

# Replacing Entries with Missing Elements with The Median
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

# Exclude the PassengerId Column
data = data.iloc[:, 1:]

print(data.head())

# Standardize Input
scaler = StandardScaler()
input = scaler.fit_transform(data)

x = data.iloc[:, 1:-1]

# Ensure PyTorch is Using our GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ' + str(device))


class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = testData(torch.FloatTensor(input))

# Initialize the Dataloaders
test_loader = DataLoader(dataset=test_data, batch_size=1)

LAYER_NODES = 6

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
model = torch.load('Titanic/trained_model')
model.to(device)

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

y_pred_list = [int(a.squeeze().tolist()) for a in y_pred_list]

# Save Our Results into The Submission DataFrame and to CSV
submission.insert(len(submission.columns), 'Survived', y_pred_list)
submission.to_csv('Titanic/Data/Submission.csv', index=False)
print(submission.head())
print(submission.shape)
