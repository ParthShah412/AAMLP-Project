from __future__ import print_function

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, Lasso, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import sklearn as sk
import pandas as pd
import numpy as np
pd.options.display.width = 200
from sklearn.model_selection import train_test_split
import random
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel

test_fraction = 0.3
RANDOM_STATE= 49
random.seed(8)
np.random.seed(8)

dfile = 'merged_file_f.csv'

def build_and_evaluate(elf_in, x_train, y_train, x_test, y_test, x, f_columns, param):
	from sklearn import metrics
	from sklearn.metrics import roc_auc_score, roc_curve
	from sklearn.model_selection import RandomizedSearchCV, KFold
	

	#print ("print x_train")
	#print (x_train)
	#y_train= y_train.ravel()
	#kf= KFold(n_splits= x_train.shape[0])
	elf_in= RandomizedSearchCV(elf_in, param_distributions= param, 
		cv= 10, n_iter= 100, random_state= RANDOM_STATE).fit(x_train, y_train)


	predictions= elf_in.predict(x_test)

	print ("accuracy: ", str(round(accuracy_score(y_test, predictions)*100, 2)), " %")
	try:	
		prob= elf_in.predict_proba(x_test)[:, 1]
	except:
		prob= elf_in.predict(x_test)
	print (classification_report(y_test, predictions))

	print (confusion_matrix(y_test, predictions))
	print ("y_test: ")
	print (y_test.sum())
	print (str(len(y_test)))
	print ("done printing y test")

	print('Feature importances:')
	print('Class {} is considered the positive class by the classifier, class {} negative'.format(elf_in.classes_[1],elf_in.classes_[0]))
	
	print ("best params")
	print (elf_in.best_params_)

	elf_in= elf_in.best_estimator_

	# try:
	# 	print (elf_in.coef_)

	# except:
	# 	pass

	try:
		top_coefs = list(elf_in.coef_[0, np.argsort(np.abs(elf_in.coef_))[:, -30:]][0])

		top_words = [f_columns[i] for i in np.argsort(np.abs(elf_in.coef_))[:, -30:][0]]

		for i in range(len(top_coefs) - 1, -1, -1):
			print('\t{}: {}'.format(top_words[i], top_coefs[i]))
	except:
		#print (list(elf.coef_[0, :][0]))
		try:
			top_coefs = list(elf_in.dual_coef_[0, np.argsort(np.abs(elf_in.dual_coef_))[:, -30:]][0])

			top_words = [f_columns[i] for i in np.argsort(np.abs(elf_in.dual_coef_))[:, -30:][0]]

			for i in range(len(top_coefs) - 1, -1, -1):
				print('\t{}: {}'.format(top_words[i], top_coefs[i]))
		except:
			pass
		pass
	fpr, tpr, thresholds= roc_curve(y_test, prob)
	print ("AUC:", str(round(metrics.auc(fpr, tpr), 5)))



def main():

	print('Building and evaluating a model using dataset: {}'.format(dfile))
	df = pd.read_csv(dfile)

	#df = sk.utils.shuffle(df)

	t_column = [c for c in df.columns if c.startswith('t_')]
	f_columns = [c for c in df.columns if c.startswith('f_')]

	#print (f_columns)

	print('Dataset has {} rows and {} feature columns'.format(df.shape[0], len(f_columns)))

	df = sk.utils.shuffle(df)
	x= df[f_columns]
	y= df[t_column]
	x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= test_fraction, random_state= 1215)
	#print ("---------------")
	#print (x_test)
	#print (y_test)
	#print (x_test)

	from sklearn.preprocessing import StandardScaler
	sc= StandardScaler()
	x_train= sc.fit_transform(x_train)
	x_test= sc.fit_transform(x_test)

	#print (x_test[0])
	#print ("done printing x_test")
	lassocv= LassoCV(eps= 0.001, n_alphas= 10, cv= 10, fit_intercept= False,
		random_state= RANDOM_STATE).fit(x_train, y_train)
	model= SelectFromModel(lassocv, prefit= True, threshold= -np.inf, max_features= 25)
	feature_idx= model.get_support()
	#print (feature_idx)
	feature_name= x.columns[feature_idx]
	print (feature_name)

	lasso_x_train= x_train
	lasso_x_test= x_test

	lasso_x_train= model.transform(x_train)
	lasso_x_test= model.transform(x_test)
	lasso_f_columns= feature_name

	ridgecv= RidgeCV(alphas= [0.01, 0.1, 1, 10], cv= 10, fit_intercept= False).fit(x_train, y_train)
	ridge_model= SelectFromModel(ridgecv, prefit= True, threshold= -np.inf, max_features= 25 )
	ridge_feature_idx= ridge_model.get_support()
	#print (feature_idx)
	ridge_feature_name= x.columns[ridge_feature_idx]
	print (feature_name)

	ridge_x_train= x_train
	ridge_x_test= x_test

	ridge_x_train= model.transform(x_train)
	ridge_x_test= model.transform(x_test)
	ridge_f_columns= feature_name

	elf_lr= LogisticRegression(max_iter= 5000, 
								solver= "liblinear")
	param_lr= {
				"C": [0.01, 0.1, 1, 10], 
			  }

	elf_lsvc= LinearSVC(dual= False)
	# elf_poly3= SVC(kernel= "poly", degree= 3).fit(x_train, y_train)
	# elf_rbf= SVC().fit(x_train, y_train)
	# elf_poly3_coef1= SVC(kernel= "poly", degree= 3, coef0= 1).fit(x_train, y_train)

	param_lsvc= {
				"penalty": ["l1", "l2"], 
				"C": [0.001, 0.01, 0.1, 1, 10],
				"max_iter": [1000, 5000],
				"fit_intercept": [True, False]
				}

	elf_svc= SVC()
	param_svc= {
				"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
				"kernel": ["rbf", "poly", "sigmoid"],
				"degree": [0, 1, 2,3,4,5,6],
				"gamma": ["scale", "auto", 0.01, 0.1, 1, 10, 100],
				"coef0": [x for x in np.arange(0.0, 5.0, 0.05)],
				"decision_function_shape": ["ovo", "ovr"]
				}

	build_and_evaluate(elf_in= elf_lr, x_train= x_train, y_train= y_train,
		x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_lr)
	print ("Done Printing LR\n")

	build_and_evaluate(elf_in= elf_lr, x_train= lasso_x_train, y_train= y_train,
		x_test=  lasso_x_test, y_test= y_test, x= x, f_columns= lasso_f_columns, param= param_lr)
	print ("Done Printing LassoCV LR\n")

	build_and_evaluate(elf_in= elf_lr, x_train= ridge_x_train, y_train= y_train,
		x_test=  ridge_x_test, y_test= y_test, x= x, f_columns= ridge_f_columns, param= param_lr)
	print ("Done Printing RidgeCV LR\n")

	build_and_evaluate(elf_in= elf_lsvc, x_train= x_train, y_train= y_train,
		x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_lsvc)
	print ("Done Printing LSVC\n")

	# build_and_evaluate(elf_poly3, x_train= x_train, y_train= y_train,
	# 	x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_svc)
	# print ("Done Printing SVM Poly3\n")

	# build_and_evaluate(elf_poly3_coef1, x_train= x_train, y_train= y_train,
	# 	x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_svc)
	# print ("Done Printing SVM Poly3 Coef 1\n")

	# build_and_evaluate(elf_rbf, x_train= x_train, y_train= y_train,
	# 	x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_svc)
	# print ("Done Printing SVM RBF\n")
	build_and_evaluate(elf_in= elf_svc, x_train= x_train, y_train= y_train,
		x_test=  x_test, y_test= y_test, x= x, f_columns= f_columns, param= param_svc)
	print ("Done Printing SVC\n")

	import time
	print (time.time())
	


if __name__ == '__main__':
	main()