# AAMLP-Project

The purpose of this project is to build a reliable model to predict volume responsiveness among Sepsis 3 patient group base on their waveform and EMR records. Lesion study is also included in order to check which category of waveform has most effect on model performance. 


--------------------MIMIC-III Data(EMR)--------------------

df5_wfflag.xlsx
ICUSTAYS.csv

----------Extract waveform----------

waveform_integration.py
waveform_integration_deprec.py

----------MIMIC-III Waveform Matched Subset----------

integrated_mac.csv
integrated_steph.csv
integrated_windows.csv

----------Waveform EDA----------

waveform_eda.py

----------Imputate Waveform Missing Data and Merge waveform with EMR data----------

either imputation_merge_data.py or imputation_merge_data.ipynb
This py file impute missing data from waveform and then merge it with EMR data using df5_wfflag.xlsx & ICUSTAYS.csv

----------Final Master Data--------------------

merged_file_f.csv
This csv file will be used for training and testing model

----------Random Forest Model----------

rf.ipynb or rf.py

----------Lasso for Logistic regression.util----------

lassocv_util.py: contain load data function and lasso feature selection 

----------Learning curve function----------

learningcurve.py: learning curve function to be used for different models

----------Lasso Logistic Regression Learning Curve----------

lassocv_lc.py

----------Model_Evaluation_ROC_learningcurve_lesionstudy.ipynb----------

Contains everything:
1. Random forest(trainng model, random search to find hyperparameter, cross-validation, ROC curve, evaluation metrices, feature importance, learning curve)
2. SVC(learning curve, hyperparameters are found from build_and_evaluate_model.py)
3. Lasso logistic regression[(training model ,hyperparameter tuning, ROC cuve, cv, evaluation metrices)also included in build_and_evaluate_model.py, lesion studies,learning curve]
4. xgBoost(training model, feature importance, ROC curve, evaluation metrices)
5. ROC curves for all models in one plot
