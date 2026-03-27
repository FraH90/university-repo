import pandas as pd
import numpy as np

# np.random.seed(123)

class DataProc:
	"""
	:param csv_path: path of the csv file containing the data we want to fit
	:param config: dictionary that contain the configuration of the linear regressor (hyperparameters, convergence condition, feature selection)
	Class to load dataset from csv file and preprocess it
	"""
	def __init__(self, csv_path, features_select, y_label, normalize=True, poly_grade=1):
		"""
		"""
		# Parameters
		self.csv_path = csv_path
		self.features_select = features_select
		self.y_label = y_label
		self.poly_grade = poly_grade
		self.normalize = normalize

		# Placeholders
		self.X, self.y = None, None
		self.X_train, self.y_train = None, None
		self.X_valid, self.y_valid = None, None
		self.X_test, self.y_test = None, None
		self.X_mean_trainingset, self.X_std_trainingset  = None, None

		# SEMI-AUTOMATIC BEHAVIOR. Automatically load data, perform preprocessing, etc
		self._load_and_preprocess()
		# Generate eventual polynomial features
		self._polynomial_features()
		# get m and n_features
		self.m_samples = self.X.shape[0]

		self._dataset_split()
		self._normalize()
		self._add_bias_column()
		self.n_features = self.X_train.shape[1]			# update n_features after adding bias column (and eventual polynomial features)
		self.m_train = self.X_train.shape[0]
		self.m_valid = self.X_valid.shape[0]
		self.m_test = self.X_test.shape[0]

	def _load_and_preprocess(self):
		'''
		This loads the data from the csv as a pandas dataframe, it select the features in which we're interested and the labels, 
		it applies shuffling to the dataset, and finally returns the data as numpy arrays
		OSS: X can be a matrix! The columns are the features, the rows the samples
		'''
		# read dataset from csv filepath
		dataset = pd.read_csv(self.csv_path)
		# shuffling all the samples to avoid group bias (the index is fixed after the shuffle by using reset_index)
		dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

		# print dataset stats
		print(dataset.describe(), '\n')
		print("Data Types:\n", dataset.dtypes)
		print()
		# test the correlation on dataset. This generates a nxn matrix (where n is the number of features)
		# and it gives a number in the range [-1, 1] (correlation coefficient) that tells statistically 
		# how that feature is correlated to another. The elements on the diagonal of this matrix will be 1
		numeric_dataset = dataset.select_dtypes(include=[np.number])
		print(numeric_dataset.corr())
		print()

		# Extract from the dataset the features specified in self.features_select, put it into X and transform it into a numpy array
		# OSS: then with the .values attribute we transform the pandas dataframes in numpy arrays!
		# THIS IS ESSENTIAL IN ORDER TO OPERATE WITH THE DATA; YOU CAN'T OPERATE ON DATAFRAME AS NUMBERS
		self.features_list = [feature.strip() for feature in self.features_select.split(',')]
		self.X = dataset[self.features_list].values
		# Extract from the dataset the label y and convert it into numpy array
		self.y = dataset[self.y_label].values
		# Let's return the processed X, y in case you need to use it outside of the class
		return self.X, self.y

	def _polynomial_features(self):
		'''
		Add polynomial features, given by poly_grade. Permitted only if the initial extracted dataset has a single feature.
		'''
		if self.poly_grade == 1:
			return
		if self.X.shape[1] != 1:
			print("Polynomial features can be retrieven only in case of single feature (X must be a vector, not a matrix)")
			print("Simple multivariate regression will be performed here, without taking into account poly features")
			return
		# Being the input univariate, we just need to extract the single feature from the dataset and compute a new self.X matrix from x
		x = self.X[:, 0]
		poly_features = [x**exp for exp in range(1, self.poly_grade)]
		self.X = np.column_stack(poly_features)

	def _dataset_split(self):
		# in order to perform hold-out splitting 80/20 identify the index at 80% of the total length of the array
		# Before train_index we'll have the training set (80%); after we'll have the remaining 20% (test set)
		train_valid_index = round(len(self.X)*0.8)
		# split the training+valid set into training set and validation, with hold-out 70/30 and get index of the validation set (it will start after the 70% of the training set)
		valid_index = round(train_valid_index*0.7)

		# split the dataset into training+valid and test set. 
		X_train_valid = self.X[:train_valid_index]
		y_train_valid = self.y[:train_valid_index]
		# ACHTUNG: don't touch the test set. Don't base the z-score normalizzation also on data present on testset!
		self.X_test = self.X[train_valid_index:]
		self.y_test = self.y[train_valid_index:]

		# split training set into training and validation. 
		# Remember: the training set is the one on which you compute the weights/parameters;
		# the validation set is the one with which you compute the error of the hypothesis (regressor) you've generated;
		# in case this fails to obtain required performances, you can select a different model (the validation set is used for model selection)
		# Finally the test set is the one on which you test the selected model, and get the final performances results (rate of false positive, false negative, etc)
		self.X_train = X_train_valid[:valid_index]
		self.y_train = y_train_valid[:valid_index]

		self.X_valid = X_train_valid[valid_index:]
		self.y_valid = y_train_valid[valid_index:]

		# Let's still return this tuple in case you need it outside of the class
		return (self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test)

	def _normalize(self):
		'''
		This method apply normalization to the feature of the datasets (train, valid, test) using z-score normalization
		Remember that z-score normalization returns datas that have zero mean and stddev=1
		'''
		if self.normalize == False:
			return
		# z-score normalization. Achtung: mean has another shape (it's a row), but by broadcasting (implicit operation)
		# In broadcasting the array "mean" here is replicated the same number of rows of X_train in order to let the arithmetic operation happen
		# Apply the same normalization to validation and test sets. THIS IS IMPORTANT! OTHERWISE PERFORMANCE METRICS WILL BE WRONG!
		# NOTICE HOW TO NORMALIZE THE FEATURES OF VALIDATION AND TEST SET WE USE THE MEAN AND STDDEV OF THE TRAINING SET
		self.X_mean_trainingset = self.X_train.mean(axis=0)
		self.X_std_trainingset = self.X_train.std(axis=0)
		_normalize_features = lambda X: (X - self.X_mean_trainingset) / self.X_std_trainingset
		################################################################################################################################################
		# ACHTUNG: The normalization must be applied to all the datasets (training, valid, test), in order to have data that span on the same scale
		# and are distributed within the same range.
		# However for the normalization, for all the datasets, you MUST use the mean and stddev of the training set, to avoid data leakage:
		# that means the model must not know any info from the validation or test set! 
		# So: NORMALIZATION IS APPLIED TO EACH DATASET (TRAIN, TEST, VALID) BUT THE MEAN AND STDDEV USED FOR THE NORMALIZATION ARE THE ONE COMPUTED 
		# FROM THE TRAINING SET
		################################################################################################################################################
		# Compute mean and std dev for each column on training set. 
		# ACHTUNG: to compute mean and stddev as column-wise opearation, you need to provide the parameter axis=0 (that means, perform the operation column-wise)
		# In numpy, axis=0 refers to the rows (so it means the mean and stddev operation scans the rows), while axis=1 refers to the columns
		# This operation makes sense since each column represent a feature, so by computing the mean of all the samples we'll have a row vector, where each element in 
		# the row represent the mean of the column of X having the same index
		# So mean, std will be both row vectors, where each element represent the mean/std of each corresponding column of X
		self.X_train = _normalize_features(self.X_train)
		self.X_valid = _normalize_features(self.X_valid)
		self.X_test = _normalize_features(self.X_test)

	def _add_bias_column(self):
		# Add bias column (of ones) to X_train, X_valid, X_test. Remember this is needed when computing h_theta(x), so theta0 is included in the summation
		# This ensure we can compute h_theta(x) on all the datasets by simply using a matrix product
		# ACHTUNG: the bias column must be added AFTER normalization
		# The c_ attribute of a nparray is to create a nparray as concatenation of columns,
		# so we generate a nparray of ones of the same dimension of X_train, and then we concatenate this to the original X_train
		add_bias = lambda X: np.c_[np.ones(X.shape[0]), X]

		# Add it also to non-normalized version, needed in order to give directly the features (ex X_valid or X_test) as input to the predictor function
		self.X_train = add_bias(self.X_train)
		self.X_valid = add_bias(self.X_valid)
		self.X_test = add_bias(self.X_test)