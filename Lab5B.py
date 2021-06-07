from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import itertools
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns

#import data
A = np.load('/home/petros/BigData/5/partB/indianpinearray.npy')
B = np.load('/home/petros/BigData/5/partB/IPgt.npy')

# print data shape and type
print(f'tc shape is {A.shape}')
print(f'gt shape is {B.shape}')
print(f'gt shape is {type(B)}')

# transform them into 1d-2d arrays
gt_array = B.reshape(B.shape[0] * B.shape[0], 1)
d_array = A.reshape(A.shape[0] * A.shape[0], 200)
print(gt_array, d_array)
data = np.concatenate((d_array, gt_array), axis=1)
print(data)

# check dimension shape and size - type
print(data.ndim, data.shape, data.size)
print("dtype", data.dtype)
print(data[-1])

# Check class distirbution
df = pd.DataFrame(data=data)
sns.countplot(df[200])

# Get rid of rows with class = 0
df = df[df[200] != 0]

|# Plot class countplot
sns.countplot(df[200])

# Create X and y from our dataframe
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = y.astype(int)
y.dtype

names = ['Alfalfa',	'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
         'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
         'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',	'Stone Steel Towers']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y)

print(X.shape, X.dtype, y.shape, y.dtype)

# StandardScaler
sns.distplot(X_train)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

sns.distplot(X_train)

# SVM

svm = SVC()

svm.get_params()


linear = SVC(kernel='linear').fit(X_train, y_train)
rbf = SVC(kernel='rbf').fit(X_train, y_train)
poly = SVC(kernel='poly').fit(X_train, y_train)
sig = SVC(kernel='sigmoid').fit(X_train, y_train)
p1 = SVC(kernel='poly', class_weight='balanced').fit(X_train, y_train)

# predictions
linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)

# Retrieve the accuracy and print it for all 4 kernel functions
# Linear
accuracy_lin = linear.score(X_test, y_test)
print('Accuracy Linear Kernel:', accuracy_lin)
cm_lin = confusion_matrix(y_test, linear_pred)
print(cm_lin)
# Polynomial
accuracy_poly = poly.score(X_test, y_test)
print('Accuracy Polynomial Kernel:', accuracy_poly)
cm_poly = confusion_matrix(y_test, poly_pred)
print(cm_poly)
# Rbf
accuracy_rbf = rbf.score(X_test, y_test)
print('Accuracy Radial Basis Kernel:', accuracy_rbf)
cm_rbf = confusion_matrix(y_test, rbf_pred)
print(cm_rbf)
# sigmoid
accuracy_sig = sig.score(X_test, y_test)
print('Accuracy Sigmoid Kernel:', accuracy_sig)

# Set the parameters by cross-validation
param_grid = {'C': [0.1, 1, 100, 1000], 'kernel': [
    'rbf'], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.score(X_test, y_test))

best_rbf = SVC(kernel='rbf', C=100, gamma=0.01).fit(X_train, y_train)
y_pred = best_rbf.predict(X_test)
accuracy_score(y_test, y_pred)

# Normalized Confusion Matrix def a function


def plot_confusion_matrix(cm, classes, normalize=True, title='Noralized Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Calculate and plot the Normalized confusion matrix
rbf_cm = confusion_matrix(y_test, y_pred)
fig = plt.figure()
fig.set_size_inches(14, 12, forward=True)

plot_confusion_matrix(rbf_cm, classes=np.asarray(
    names), normalize=True, title='Normalized confusion matrix')

print(classification_report(y_test, y_pred, target_names=names))


# RANDOM FOREST
rfc = RandomForestClassifier(random_state=42)
rfc.get_params()

param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)

CV_rfc.best_params_
print(CV_rfc.score(X_test, y_test))
best_rfc = RandomForestClassifier(random_state=42, max_features='auto',
                                  n_estimators=200, max_depth=8, criterion='entropy')

best_rfc.fit(X_train, y_train)
y_rfc = best_rfc.predict(X_test)
accuracy_score(y_test, y_rfc)

# Norm CM
rfc_cm = confusion_matrix(y_test, y_rfc)
fig = plt.figure()
fig.set_size_inches(14, 12, forward=True)

plot_confusion_matrix(rfc_cm, classes=np.asarray(
    names), normalize=True, title='Normalized confusion matrix')
print(classification_report(y_test, y_rfc, target_names=names))

rf = RandomForestClassifier(random_state=42, max_features='auto',
                            n_estimators=200, max_depth=100, criterion='entropy')

rf.get_params()
rf.fit(X_train, y_train)
pred1 = rf.predict(X_test)
print(classification_report(y_test, pred1, target_names=names))

#Pytorch

import torch.nn  as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,16)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = F.log_softmax(X, dim=1)
        return X

#'data': torch.from_numpy(self.data[idx]).type(torch.float32),
#'labels': torch.Tensor(self.labels[idx]).type(torch.int64)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#import torch
#torch.cuda.is_available()

#Import data
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self,data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self,idx):
        d = {
            'data': self.data[idx],
            'labels': self.labels[idx]
        }
        return d
X_train.dtype
y_train.dtype

train_dataset = CustomDataset(torch.from_numpy(X_train), torch.LongTensor(y_train))
print(len(train_dataset))
print(train_dataset[4])

print(X_train.shape)
print(X_train[:5])
test_dataset = CustomDataset(torch.from_numpy(X_test), torch.LongTensor(y_test))

trainloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

y
y.min()
y.max()
y.dtype

### TRAIN YOUR MODEL

import torch.optim as optim

net = MyNet()
num_epoch = 5

criterion = nn.NLLLoss()

optimizer = optim.SGD(net.parameters(), lr=0.002)

y
y1 = batch['labels'].view(1,-1)

for epoch in range(num_epoch):
    total_loss = 0
    for batch in trainloader:
        optimizer.zero_grad()

        X = batch['data']
        y = batch['labels'].view(-1)

        y_pred = net(X.float())

        loss = criterion(y_pred, y)
        total_loss += loss
        loss.backward()

        optimizer.step()
    print(f'Epoch {epoch+1} done training! w/ loss {total_loss}')

    total = 0
    correct = 0

    for batch in testloader:
        X = batch['data']
        y = batch['labels'].view(-1)
        y_pred = net(X)

        total += float(y.size()[0])
        correct += float((torch.argmax(y_pred, axis=1) == y).sum())

    print(f'test accuracy: {100*correct/total:2.2f}%')
