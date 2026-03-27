import numpy as np
import pandas as pd

from neural_network_classification import NeuralNet

heart_df = pd.read_csv('./data/heart.csv')
print(heart_df.shape)

print(heart_df.isna().sum())

print(heart_df.dtypes)

# shuffling all samples to avoid group bias
index = heart_df.index
heart_df = heart_df.iloc[np.random.choice(index, len(index))]


#convert imput to numpy arrays
X = heart_df.drop(columns=['disease']).values

#replace target class with 0 and 1
#1 means "have heart disease" and 0 means "do not have heart disease"
heart_df['disease'] = heart_df['disease'].replace('n', 0)
heart_df['disease'] = heart_df['disease'].replace('y', 1)

y_label = heart_df['disease'].values.reshape(X.shape[0], 1)

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

nn = NeuralNet(layers=[X_train.shape[1], 50, 50, 1], learning_rate=0.01, iterations=1000, lmd=0.001)
nn.fit(X_train, y_label_train)

nn.plot_loss()

test_pred = nn.predict(X_test)
print("Test accuracy is {}".format(nn.acc(y_label_test.reshape(-1), test_pred.reshape(-1))))