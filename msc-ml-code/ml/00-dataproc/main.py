from dataproc import DataProc

# Load data and preprocess it
data = DataProc('fake_dataset.csv', 'x1, x2, x5', 'y')
print(data.X_train)

# Load data and preprocess it
data = DataProc('fake_dataset.csv', 'x1', 'y', poly_grade=3)
print(data.X_train)