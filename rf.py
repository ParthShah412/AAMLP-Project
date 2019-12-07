# Import neccessary libries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 283)

from sklearn import preprocessing
import matplotlib.pyplot as plt # For ploting 
import seaborn as sns
plt.rc("font", size=14)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, RandomizedSearchCV
from sklearn import metrics

from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_curve, SCORERS
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle

def load_data(file):
	# Read in dataset
	df = pd.read_csv('merged_file_f.csv')
	t_column = [c for c in df.columns if c.startswith('t_')]
	f_columns = [c for c in df.columns if c.startswith('f_')]

	print('Dataset has {} rows and {} feature columns'.format(df.shape[0], len(f_columns)))
	import random
	random.seed(8)
	np.random.seed(8)
	df = shuffle(df)
	x= df[f_columns]
	y= df[t_column]

	return x,y,df

def train(X_train, y_train, paramdict):

	rf = RandomForestClassifier(random_state = 49)
	rf_random = RandomizedSearchCV(rf, 
		param_distributions= paramdict,
		random_state = 49,
		cv = 10,
		n_iter = 100)

	# fit the random search model
	rf_random.fit(X_train, y_train.values.ravel())

	# Best random search parameters 
	print("-----Best random search parameters-----")
	print(rf_random.best_params_)
	best_random = rf_random.best_estimator_

	# Print top 30 important features in this model
	print ("\n-----Feature Importance-----")
	feature_imp = pd.Series(best_random.feature_importances_,index=x.columns).sort_values(ascending=False)
	print(feature_imp.head(n=30))

	return best_random


def predict(best_model, X_valid):
	y_pred = best_model.predict(X_valid)
	y_prob = best_model.predict_proba(X_valid)[:,1]

	return y_pred, y_prob

def evaluate(y_valid, y_pred, y_prob):
	print("\n-----Classification Report-----")
	print(classification_report(y_test.values.ravel(), y_pred))


	print("\n-----Accuracy-----")
	print(accuracy_score(y_test.values.ravel(), y_pred))

	fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), y_prob)
	plt.figure()
	plt.plot(fpr, tpr, label='Random Forest')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.savefig('ROC')
	plt.show()
	print ( 'Random Forest AUC: ' + str(round(metrics.auc(fpr,tpr),5)))
	print('Random Forest F1 Score: ' + str(round(f1_score(y_test.values.ravel(), y_pred),5)))

def test(best_model, X_valid, y_valid):
	y_pred, y_prob = predict(best_model, X_valid)
	evaluate(y_valid, y_pred, y_prob)


if __name__ == '__main__':

	x, y, df = load_data('merged_file_f.csv')
	# Split data into training and testing set
	# 70% training and 30% testing 
	x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state = 1215)


	paramdict = dict(criterion = ['gini','entropy'],
                 # Method of selecting samples for training each tree
                 bootstrap = [True, False],
                 # Number of trees in random forest
                 n_estimators = list(range(5,51,5)),
                 # Maximum number of levels in tree
                 max_depth = list(range(1,11)),
                 # Minimum number of samples required to split a node
                 min_samples_split = list(range(2,11)),
                 # Minimum number of samples required to be at a leaf node
                 min_samples_leaf = list(range(1,16)))

	best_model = train(x_train,y_train,paramdict)
	test(best_model, x_test, y_test)
