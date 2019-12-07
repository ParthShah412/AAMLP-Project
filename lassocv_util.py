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
from sklearn.utils import shuffle

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


def lasso_cv_x(x_train, x_test, y_train, y_test):
    
    x,y,df = load_data('merged_file_f.csv')
    # Split data into training and testing set
    # 70% training and 30% testing 
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state = 1215) 

    # Standard Scaling X
    from sklearn.preprocessing import StandardScaler
    sc= StandardScaler()
    x_train= sc.fit_transform(x_train)
    x_test= sc.fit_transform(x_test)

    RANDOM_STATE = 49
    # Get LassoCV of X
    lassocv= LassoCV(eps= 0.001, n_alphas= 10, cv= 10, fit_intercept= False,
        random_state= RANDOM_STATE).fit(x_train, y_train)
    model= SelectFromModel(lassocv, prefit= True, threshold= -np.inf, max_features= 25)
    
    feature_idx= model.get_support()
    feature_name= x.columns[feature_idx]

    lasso_x_train= model.transform(x_train)
    lasso_x_test= model.transform(x_test)
    lasso_f_columns= feature_name
    
    return lasso_x_train, lasso_x_test, lasso_f_columns