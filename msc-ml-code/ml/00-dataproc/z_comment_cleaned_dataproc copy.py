import pandas as pd
import numpy as np

# np.random.seed(123)

class DataProc:
	def __init__(self, csv_path, features_select, y_label, normalize=True, poly_grade=1):
		self.csv_path = csv_path
		self.features_select = features_select
		self.y_label = y_label
		self.poly_grade = poly_grade
		self.normalize = normalize

		self.X, self.y = None, None
		self.X_train, self.y_train = None, None
		self.X_valid, self.y_valid = None, None
		self.X_test, self.y_test = None, None
		self.X_mean_trainingset, self.X_std_trainingset  = None, None

		self._load_and_preprocess()
		self._polynomial_features()

		self.m_samples = self.X.shape[0]

		self._dataset_split()
		self._normalize()
		self._add_bias_column()
		self.n_features = self.X_train.shape[1]
		self.m_train = self.X_train.shape[0]
		self.m_valid = self.X_valid.shape[0]
		self.m_test = self.X_test.shape[0]

	def _load_and_preprocess(self):
		dataset = pd.read_csv(self.csv_path)
		dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

		# print dataset stats
		print(dataset.describe(), '\n')
		print("Data Types:\n", dataset.dtypes)
		print()
		numeric_dataset = dataset.select_dtypes(include=[np.number])
		print(numeric_dataset.corr())
		print()

		self.features_list = [feature.strip() for feature in self.features_select.split(',')]
		self.X = dataset[self.features_list].values
		self.y = dataset[self.y_label].values
		return self.X, self.y

	def _polynomial_features(self):
		if self.poly_grade == 1:
			return
		if self.X.shape[1] != 1:
			print("Polynomial features can be retrieven only in case of single feature (X must be a vector, not a matrix)")
			print("Simple multivariate regression will be performed here, without taking into account poly features")
			return
		x = self.X[:, 0]
		poly_features = [x**exp for exp in range(1, self.poly_grade)]
		self.X = np.column_stack(poly_features)

	def _dataset_split(self):
		train_valid_index = round(len(self.X)*0.8)
		valid_index = round(train_valid_index*0.7)
		X_train_valid = self.X[:train_valid_index]
		y_train_valid = self.y[:train_valid_index]
		self.X_test = self.X[train_valid_index:]
		self.y_test = self.y[train_valid_index:]
		self.X_train = X_train_valid[:valid_index]
		self.y_train = y_train_valid[:valid_index]
		self.X_valid = X_train_valid[valid_index:]
		self.y_valid = y_train_valid[valid_index:]
		return (self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test)

	def _normalize(self):
		if self.normalize == False:
			return
		self.X_mean_trainingset = self.X_train.mean(axis=0)
		self.X_std_trainingset = self.X_train.std(axis=0)
		_normalize_features = lambda X: (X - self.X_mean_trainingset) / self.X_std_trainingset
		self.X_train = _normalize_features(self.X_train)
		self.X_valid = _normalize_features(self.X_valid)
		self.X_test = _normalize_features(self.X_test)

	def _add_bias_column(self):
		add_bias = lambda X: np.c_[np.ones(X.shape[0]), X]
		self.X_train = add_bias(self.X_train)
		self.X_valid = add_bias(self.X_valid)
		self.X_test = add_bias(self.X_test)