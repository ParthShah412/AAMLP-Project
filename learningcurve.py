from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # For ploting 


def plot_learning_curve(clf, x_train, y_train, model_name):
    ''' Plot the learning curve of a given classifier.
    '''
    cv = 5
    y_train = np.array(y_train).reshape(y_train.size)
    n_samples = y_train.size *(1-1/cv)
    plot_per_k_samples = 10  # plot once every k samples
    train_sizes = np.linspace(0.1,1,30)
    
    
    train_sizes, train_scores, validation_scores = learning_curve(
            estimator = clf,
            X = x_train, y = y_train, 
            train_sizes = train_sizes, 
            cv = 5)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)

    
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training score')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation score')
    plt.ylabel('Score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning Curves for a {} Model'.format(model_name), fontsize = 18, y = 1.03)
    plt.legend()
    #     plt.ylim(0,1.05)   
    plt.show()
    
    return None