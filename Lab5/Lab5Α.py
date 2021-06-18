import numpy as np
import pandas as pd
import seaborn as sns
import scipy

from PIL import Image
from matplotlib import pyplot as plt

#Insert our data

green = Image.open('/home/petros/BigData/5/partA/green.tif')
nir = Image.open('/home/petros/BigData/5/partA/nir.tif')
gt = Image.open('/home/petros/BigData/5/partA/gt.tif')

green_array = np.array(green)
nir_array = np.array(nir)
gt_array = np.array(gt)

green_array.shape
nir_array.shape
gt_array.shape

green_array = green_array.reshape(green_array.shape[0] * green_array.shape[0], 1)
nir_array = nir_array.reshape(nir_array.shape[0] * nir_array.shape[0], 1)
gt_array = gt_array.reshape(gt_array.shape[0] * gt_array.shape[0], 1)

data_array = np.concatenate((green_array, nir_array, gt_array), axis=1)
data_array.shape

train_df = pd.DataFrame(data=data_array, index=range(green_array.shape[0]), columns = ['green', 'nir', 'gt'])
train_df
train_df['gt'] = train_df['gt'].replace(255,1)
train_df['gt']

sns.set(rc={'figure.figsize': (6,6)})
sns.countplot(train_df['gt'])


f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(train_df['green'])

f.add_subplot(1,2,2)
sns.distplot(train_df['nir'])
##

X = train_df[['green','nir']]
y = train_df['gt']

print(X.shape, y.shape)

#SKLEARN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
#Training the model
gnb.fit(X_train, y_train)
#Predict testing set
y_pred = gnb.predict(X_test)
#Check performance of model
print(accuracy_score(y_test, y_pred))
print(f'Number of mislabeled points out of a total {X_test.shape[0]} point: {(y_pred != y_test).sum()}')
labels = ['Χερσαίο Έδαφος', 'Υδάτινο σώμα']
classification_report(y_test,y_pred,target_names=labels)
print(classification_report(y_test,y_pred,target_names=labels))
c1 = confusion_matrix(y_test, y_pred)
print(c1)


gnb.class_prior_
gnb.get_params()

#custom prior
mn = GaussianNB(priors=[0.3,0.7])
y_pred_1 = mn.fit(X_train, y_train).predict(X_test)
print(accuracy_score(y_test, y_pred_1))
print(f'Number of mislabeled points out of a total {X_test.shape[0]} point: {(y_pred_1 != y_test).sum()}')
c1 = confusion_matrix(y_test, y_pred_1)
print(c1)

mn.class_prior_
mn.get_params()

ml = GaussianNB(var_smoothing=1e-3)
y_pred_2 = ml.fit(X_train, y_train).predict(X_test)
print(accuracy_score(y_test, y_pred_2))
print(f'Number of mislabeled points out of a total {X_test.shape[0]} point: {(y_pred_2 != y_test).sum()}')
results = confusion_matrix(y_test,y_pred)
print( 'confusion matrix :')
print(results)
accuracy_score(y_test,y_pred)
print('accuracy_score: ')
print(accuracy_score)
print(classification_report(y_test,y_pred))


#KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print(f'Mistakes {(y_pred != y_test).sum()} out of {len(y_test)}')
print(accuracy_score(y_test,y_pred))
labels = ['Χερσαίο Έδαφος', 'Υδάτινο σώμα']
print(classification_report(y_test,y_pred,target_names=labels))

c1 = confusion_matrix(y_test, y_pred)
print(c1)

# Manhattan

knn1 = KNeighborsClassifier(p=1, weights='distance')
knn1.fit(X_train,y_train)
y_pred = knn1.predict(X_test)

print(f'Mistakes {(y_pred != y_test).sum()} out of {len(y_test)}')
print(accuracy_score(y_test,y_pred))
labels = ['Χερσαίο Έδαφος', 'Υδάτινο σώμα']
print(classification_report(y_test,y_pred,target_names=labels))

c1 = confusion_matrix(y_test, y_pred)
print(c1)


#K Number
knn2 = KNeighborsClassifier(n_neighbors=10)
knn2.fit(X_train, y_train)
y_pred = knn2.predict(X_test)

print(f'Mistakes {(y_pred != y_test).sum()} out of {len(y_test)}')
print(accuracy_score(y_test,y_pred))
labels = ['Χερσαίο Έδαφος', 'Υδάτινο σώμα']
print(classification_report(y_test,y_pred,target_names=labels))

c1 = confusion_matrix(y_test, y_pred)
print(c1)

#Preprocess The X Data By Scaling

# Train the scaler, which standarizes all the features to have mean=0 and unit variance
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
# Apply the scaler to the X training data
X_train = sc.transform(X_train)
# Apply the same scaler to the X test data
X_test = sc.transform(X_test)

print(X_train,X_test)
sns.distplot(X_train)
#Perceptron

from sklearn.linear_model import Perceptron
perc = Perceptron(random_state=21)
perc.fit(X_train, y_train)
y_pred = perc.predict(X_test)
print(f'Number of mislabeled points out of a total {X_test.shape[0]} points: {(y_pred != y_test).sum()}')

perc = Perceptron(max_iter=50, eta0=0.0001, tol=1e-6, random_state=21)
perc.fit(X_train, y_train)
y_pred = perc.predict(X_test)
print(f'Number of mislabeled points out of a total {X_test.shape[0]} points: {(y_pred != y_test).sum()}')
print(classification_report(y_test, y_pred))
