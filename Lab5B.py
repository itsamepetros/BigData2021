from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import itertools
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
df
# Get rid of rows with class = 0
df = df[df[200] != 0]
df
|# Plot class countplot
sns.countplot(df[200])

# Create X and y from our dataframe
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
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
X_train.shape


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

# Create X and y from our dataframe
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y.dtype
y=y-1
y = y.astype(int)
y.min()
names = ['Alfalfa',	'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees',
         'Grass-pasture-mowed', 'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
         'Soybean-clean', 'Wheat', 'Woods', 'Buildings Grass Trees Drives',	'Stone Steel Towers']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21, stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X.shape, X.dtype, y.shape, y.dtype)
X


import torch.nn  as nn
import torch.nn.functional as F

#1
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,16)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)
        #X = F.log_softmax(X, dim=1)
        return X

#2

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = F.log_softmax(X, dim=1)
        return X



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

training_data = CustomDataset(torch.from_numpy(X_train), torch.LongTensor(y_train))
test_data = CustomDataset(torch.from_numpy(X_test), torch.LongTensor(y_test))


trainloader = DataLoader(training_data, batch_size=50, shuffle=True)
testloader = DataLoader(test_data, batch_size=200, shuffle=False)

### train MyNet model

import torch.optim as optim

net = MyNet()
learning_rate = 1e-3
num_epoch = 200

net.parameters

optimizer = torch.optim.SGD(net.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()



# Initialize losses & accuracies of train and test set
train_losses = list()
test_losses = list()
train_acc = list()
test_acc = list()

for epoch in range(num_epoch):

    # Initialization for "trainloader loop"
    total_loss = 0
    total = 0
    correct = 0
    #Trainset loop
    for batch in trainloader:
        #set the gradients to zero
        optimizer.zero_grad()
        #Data into batches
        X = batch['data']
        y = batch['labels'].view(-1)
        #Model Prediction
        y_pred = net(X.float())
        # Calculate num of total y and correct predictions
        total += float(y.size()[0])
        correct += float((torch.argmax(y_pred, axis=1) == y).sum())
        #Calculate train loss
        loss = criterion(y_pred, y)
        #backpropagation
        loss.backward()
        # Perform one step of optimization method
        optimizer.step()
        # Append loss of each loss to total_loss
        total_loss += loss.detach()
      # Evaluate train loss & accuracy per epoch
    train_losses.append(total_loss/len(trainloader))
    train_acc.append(correct/total)

    # Initialization for "testloader loop"
    eval_loss = 0
    total = 0
    correct = 0
    #Testset loop
    for batch in testloader:
        #Data into batches
        X = batch['data']
        y = batch['labels'].view(-1)
        #Model Prediction
        y_pred = net(X.float())
        # Calculate num of total y and correct predictions
        total += float(y.size()[0])
        correct += float((torch.argmax(y_pred, axis=1) == y).sum())
        # Calculate evaluation loss
        loss = criterion(y_pred, y)
        eval_loss += loss

      # Evaluate train loss & accuracy per epoch!
    test_losses.append(eval_loss/len(testloader))
    test_acc.append(correct/total)

    print(f'Epoch {epoch+1} completed. Train_loss:{total_loss/len(trainloader)}, Test_loss:{eval_loss/len(testloader)}, acc:{100*correct/total:02.2f}%')

#Plot Learning Curves
#train test loss

fig = plt.figure(figsize=(20,10))
plt.title('Train - Test Loss')
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.xlabel('num_epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')

#Tran Val Accuracy

fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')

#Evaluate model
y_pred_list =[]
with torch.no_grad():
    net.eval()
    for batch in testloader:
        X = batch['data']
        y_test_pred = net(X.float())
        y_pred_tags = torch.argmax(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

y_net = []
for sublist in y_pred_list:
    for item in sublist:
        y_net.append(item)

#Confusion Matrix Dataframe
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_net))
confusion_matrix_df.columns = names

#Plot Confusion Matri
plt.figure(figsize = (10,10))
sns.set(font_scale=1)
sns_plot = sns.heatmap(confusion_matrix_df, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)
print(classification_report(y_test, flat_list, target_names=names))

#2

net = MLP()
learning_rate = 0.003
num_epoch = 200

net.parameters

optimizer = torch.optim.SGD(net.parameters(), learning_rate)
criterion = nn.NLLLoss()

# Initialize losses & accuracies of train and test set
train_losses = list()
test_losses = list()
train_acc = list()
test_acc = list()

for epoch in range(num_epoch):

    # Initialization for "trainloader loop"
    total_loss = 0
    total = 0
    correct = 0
    #Trainset loop
    for batch in trainloader:
        #set the gradients to zero
        optimizer.zero_grad()
        #Data into batches
        X = batch['data']
        y = batch['labels'].view(-1)
        #Model Prediction
        y_pred = net(X.float())
        # Calculate num of total y and correct predictions
        total += float(y.size()[0])
        correct += float((torch.argmax(y_pred, axis=1) == y).sum())
        #Calculate train loss
        loss = criterion(y_pred, y)
        #backpropagation
        loss.backward()
        # Perform one step of optimization method
        optimizer.step()
        # Append loss of each loss to total_loss
        total_loss += loss.detach()
      # Evaluate train loss & accuracy per epoch
    train_losses.append(total_loss/len(trainloader))
    train_acc.append(correct/total)

    # Initialization for "testloader loop"
    eval_loss = 0
    total = 0
    correct = 0
    #Testset loop
    for batch in testloader:
        #Data into batches
        X = batch['data']
        y = batch['labels'].view(-1)
        #Model Prediction
        y_pred = net(X.float())
        # Calculate num of total y and correct predictions
        total += float(y.size()[0])
        correct += float((torch.argmax(y_pred, axis=1) == y).sum())
        # Calculate evaluation loss
        loss = criterion(y_pred, y)
        eval_loss += loss

      # Evaluate train loss & accuracy per epoch!
    test_losses.append(eval_loss/len(testloader))
    test_acc.append(correct/total)

    print(f'Epoch {epoch+1} completed. Train_loss:{total_loss/len(trainloader)}, Test_loss:{eval_loss/len(testloader)}, acc:{100*correct/total:02.2f}%')

#Plot Learning Curves
#train test loss

fig = plt.figure(figsize=(20,10))
plt.title('Train - Test Loss')
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.xlabel('num_epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend(loc='best')

#Tran Val Accuracy

fig = plt.figure(figsize=(20,10))
plt.title("Train - Validation Accuracy")
plt.plot(train_acc, label='train')
plt.plot(test_acc, label='test')
plt.xlabel('num_epochs', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.legend(loc='best')

#Evaluate model
y_pred_list =[]
with torch.no_grad():
    net.eval()
    for batch in testloader:
        X = batch['data']
        y_test_pred = net(X.float())
        y_pred_tags = torch.argmax(y_test_pred, dim=1)
        y_pred_list.append(y_pred_tags.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

y_net = []
for sublist in y_pred_list:
    for item in sublist:
        y_net.append(item)

#Confusion Matrix Dataframe
confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_net))
confusion_matrix_df.columns = names

#Plot Confusion Matri
plt.figure(figsize = (10,10))
sns.set(font_scale=1)
sns_plot = sns.heatmap(confusion_matrix_df, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)

print(classification_report(y_test, y_net, target_names=names))


#MODEL 3
#Pytorch

import torch.nn  as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim

#TRAIN VALIDATION TEST SET

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

#NORMALIZE INPUT

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

def get_class_distribution(obj):
    count_dict = {
        "class_1": 0,
        "class_2": 0,
        "class_3": 0,
        "class_4": 0,
        "class_5": 0,
        "class_6": 0,
        "class_7": 0,
        "class_8": 0,
        "class_9": 0,
        "class_10": 0,
        "class_11": 0,
        "class_12": 0,
        "class_13": 0,
        "class_14": 0,
        "class_15": 0,
        "class_16": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['class_1'] += 1
        elif i == 1:
            count_dict['class_2'] += 1
        elif i == 2:
            count_dict['class_3'] += 1
        elif i == 3:
            count_dict['class_4'] += 1
        elif i == 4:
            count_dict['class_5'] += 1
        elif i == 5:
            count_dict['class_6'] += 1
        elif i == 6:
            count_dict['class_7'] += 1
        elif i == 7:
            count_dict['class_8'] += 1
        elif i == 8:
            count_dict['class_9'] += 1
        elif i == 9:
            count_dict['class_10'] += 1
        elif i == 10:
            count_dict['class_11'] += 1
        elif i == 11:
            count_dict['class_12'] += 1
        elif i == 12:
            count_dict['class_13'] += 1
        elif i == 13:
            count_dict['class_14'] += 1
        elif i == 14:
            count_dict['class_15'] += 1
        elif i == 15:
            count_dict['class_16'] += 1
        else:
            print("Check classes.")

    return count_dict

#CUSTOM DATASET

class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())


#WEIGHTED SAMPLING

target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]

class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float)
print(class_weights)

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class_weights_all = class_weights[target_list]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

#Model Parameters

epochs = 200
batch_size = 30
learning_rate = 0.0007

num_features = 200
num_classes = 16

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)



class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


model = MulticlassClassification(num_feature = num_features, num_class=num_classes)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(model)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}



from tqdm import tqdm as tqdm




for e in tqdm(range(1, epochs+1)):

    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch, y_train_batch
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()


    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch, y_val_batch

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))


    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


y_pred_list = []

with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch
        y_test_pred = model(X_batch)
        y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list))
confusion_matrix_df.columns = names

plt.figure(figsize = (10,10))
sns.set(font_scale=1)
sns_plot = sns.heatmap(confusion_matrix_df, cmap = 'Blues', annot = True, fmt='g', vmin=0, vmax=200)

print(classification_report(y_test, y_pred_list, target_names=names))
