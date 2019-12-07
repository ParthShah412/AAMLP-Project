from lassocv_util import lasso_cv_x, load_data
from learningcurve import plot_learning_curve
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LassoCV, Lasso, RidgeCV
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
	elf_lr= LogisticRegression(max_iter= 5000,
							solver= "liblinear",
							C= 1)

	x,y,df = load_data('merged_file_f.csv')
	x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state = 1215) 

	lasso_x_train, lasso_x_test, lasso_f_columns = lasso_cv_x(x_train, x_test, y_train, y_test)
	plot_learning_curve(elf_lr, lasso_x_train, y_train, model_name='Logistic Regression')