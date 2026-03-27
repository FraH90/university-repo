import numpy as np
import pandas as pd

from neural_network_regression import NeuralNet

wine = pd.read_csv('./data/wine.csv', header=None)
print(wine.shape)

print(wine.isna().sum())

print(wine.dtypes)

# shuffling all samples to avoid group bias
# shuffling all samples to avoid group bias
index = wine.index
wine = wine.iloc[np.random.choice(index, len(index))]

#convert imput to numpy arrays
X = wine.values[:, :-1]


y_label = wine.values[:, -1].reshape(X.shape[0], 1)

# in order to perform hold-out splitting 80/20 identify max train index value
train_index = round(len(X) * 0.8)

# split dataset into training and test
X_train = X[:train_index]
y_label_train = y_label[:train_index]

X_test = X[train_index:]
y_label_test = y_label[train_index:]

# compute mean and standard deviation ONLY ON TRAINING SAMPLES
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

# apply mean and std (standard deviation) compute on training sample to training set and to test set
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print(f"Shape of train set is {X_train.shape}")
print(f"Shape of test set is {X_test.shape}")
print(f"Shape of train label is {y_label_train.shape}")
print(f"Shape of test labels is {y_label_test.shape}")

nn = NeuralNet(layers=[X_train.shape[1], 25, 25, 1], learning_rate=0.5, iterations=1000, lmd=0)
nn.fit(X_train, y_label_train)

nn.plot_loss()

test_pred = nn.predict(X_test)
print("Test error (RMSE) is {}".format(nn.rmse(y_label_test.reshape(-1), test_pred.reshape(-1))))