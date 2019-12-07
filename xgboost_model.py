import warnings
warnings.filterwarnings('ignore')

# Import neccessary libries
import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt # For ploting 

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, accuracy_score, average_precision_score, precision_recall_curve, SCORERS, f1_score, precision_score, recall_score

from sklearn import metrics

# Read in dataset
def load_data(file):
	# Read in dataset
	df = pd.read_csv('merged_file_f.csv')
	t_column = [c for c in df.columns if c.startswith('t_')]
	f_columns = [c for c in df.columns if c.startswith('f_')]
	import random
	random.seed(8)
	np.random.seed(8)

	df = shuffle(df)
	x= df[f_columns]
	y= df[t_column]

	return x,y,df



# Use XGBoost
def xgboost_train(X,Y):
	from xgboost import XGBClassifier
	from xgboost import plot_importance
	from sklearn.model_selection import train_test_split
	from sklearn.feature_selection import SelectFromModel
	from sklearn.metrics import accuracy_score

	# splitting data in test train 
	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.3, random_state=1215)
	
	# creating model 
	model = XGBClassifier(random_state = 49)
	
	# fitting model 
	model.fit(X_train,Y_train)

	return model, X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
	x,y,df = load_data('merged_file_f.csv')
	xg_rf, xg_x_train, xg_x_test, xg_y_train, xg_y_test = xgboost_train(x,y.values.ravel())

	xg_y_pred = xg_rf.predict(xg_x_test)
	xg_y_prob = xg_rf.predict_proba(xg_x_test)[:,1]

	print("\n-----Classification Report-----")
	print(classification_report(xg_y_test, xg_y_pred))

	print ("\n-----Feature Importance-----")
	xg_feature_imp = pd.Series(xg_rf.feature_importances_,index=x.columns).sort_values(ascending=False)
	print(xg_feature_imp.head(n=10))

	print("\n-----Accuracy-----")
	print(accuracy_score(xg_y_test, xg_y_pred))

	fpr, tpr, thresholds = roc_curve(xg_y_test, xg_y_prob)
	plt.figure()
	plt.plot(fpr, tpr, label='XGBoost')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.savefig('XG Boost ROC')
	plt.show()
	print ( 'AUC: ' + str(round(metrics.auc(fpr,tpr),5)))
	print('F1 Score: ' + str(round(f1_score(xg_y_test, xg_y_pred),5)))

