# Import neccessary libries
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 222)
import matplotlib
import missingno as msno


# Read data 
df_1 = pd.read_csv('integrated_mac.csv')
df_2 = pd.read_csv('integrated_steph.csv')
df_3 = pd.read_csv('integrated_windows.csv')

merged_df = pd.concat([df_1, df_2, df_3])
# Check the shape of new df
# merged_df.shape

# Plot missing value matrix
missingdata_df = merged_df.columns[merged_df.isnull().any()].tolist()
msno.matrix(merged_df[missingdata_df])

# Plot correlation heatmap
msno.heatmap(merged_df[missingdata_df], figsize = (20,20))
# This map describes the degree of nullity relationship between the different features
# The range of this nullity correlationis from -1 <= R <= 1. 
# If the nullity correlation is between -0.05 to 0.05, no value will be displayed.
# A perfect positive nullity correlation R = 1 indicates when the first feature and second feature both have corresponding missing values while a perfect negative nullity correlation(R= -1) means that one of the features is missing and the second is not missing. 

merged_df.describe()


# There are 496 out of 524 records have lead II waveform information. 
# Same = Extracted_length(), II_autocorrelation_lag_0(1), PLETH__autocorrelation__lag_0(1)
# 
# There are 352 out of 524 records have PLETH waveform information. 
# 
# 
# All 0s: II__large_standard_deviation__r_0.4, II__large_standard_deviation__r_0.45, II__large_standard_deviation__r_0.5, II__large_standard_deviation__r_0.55, II__large_standard_deviation__r_0.6000000000000001, II__large_standard_deviation__r_0.65, II__large_standard_deviation__r_0.7000000000000001, II__large_standard_deviation__r_0.75, II__large_standard_deviation__r_0.8, II__large_standard_deviation__r_0.8500000000000001, II__large_standard_deviation__r_0.9, II__large_standard_deviation__r_0.9500000000000001	
# 
# 
# All 0s: V__large_standard_deviation__r_0.4	V__large_standard_deviation__r_0.45	V__large_standard_deviation__r_0.5	V__large_standard_deviation__r_0.55	V__large_standard_deviation__r_0.6000000000000001	V__large_standard_deviation__r_0.65	V__large_standard_deviation__r_0.7000000000000001	V__large_standard_deviation__r_0.75	V__large_standard_deviation__r_0.8	V__large_standard_deviation__r_0.8500000000000001	V__large_standard_deviation__r_0.9	V__large_standard_deviation__r_0.9500000000000001
# 
# 
# 
# 
# II_first_location_of_maximum, II_first_location_of_minimum, II_large_standard_deviation_r_0.05, II_large_standard_deviation_r_0.01, II__large_standard_deviation__r_0.15000000000000002, II__large_standard_deviation__r_0.2, II__large_standard_deviation__r_0.25, II__large_standard_deviation__r_0.30000000000000004, II__large_standard_deviation__r_0.35000000000000003, II__large_standard_deviation__r_0.4, 
# 
# II_large_standard_deviation_r_0.01, II__large_standard_deviation__r_0.15000000000000002, II__large_standard_deviation__r_0.2 = 25% = 0 
# 
# II__large_standard_deviation__r_0.15000000000000002, II__large_standard_deviation__r_0.2 = 50% = 0 
# 

# Conclude that the missingless values in this dataset is missing completely at random, so we will use MICE techniques to do the imputation.


# Import imputation library
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  

mice_imputer = IterativeImputer(max_iter = 5, random_state = 0)
merged_df_mice_imputed = merged_df.copy()
# Slice TFLUID from merged_df 
merged_df_mice_imputed = merged_df_mice_imputed.loc[:,merged_df_mice_imputed.columns != "TFLUID"]
merged_df_mice_imputed.iloc[:,:] = mice_imputer.fit_transform(merged_df_mice_imputed)

# Check imputation 
merged_df_mice_imputed.describe()

merged_df_mice_imputed['TFLUID'] = merged_df['TFLUID']


# Merge current imputed waveform data with EMR data

# Import icustays.csv to know icustay_id for each subject_id
# Import df5_wfflag.xlsx to join with waveform data
# subject_id may have multiple icustay_id 
icustays = pd.read_csv("ICUSTAYS.csv")
emrdata = pd.read_excel("df5_wfflag.xlsx")


# Delete the first useless column from merged_df
merged_df_mice_imputed = merged_df_mice_imputed.drop(['Unnamed: 0'],axis=1)

emrdata.columns = emrdata.columns.str.upper()


# Merge icustays with emrdata to get subject_id
combined_df_1 = pd.merge(icustays[['SUBJECT_ID','ICUSTAY_ID','INTIME','OUTTIME']],
                        merged_df_mice_imputed,
                       on = 'SUBJECT_ID')
combined_df_1 = combined_df_1.dropna()

# Check if fluid administration time fall in icu stay time range
combined_df_1['TFLUID'] = pd.to_datetime(combined_df_1['TFLUID'])
combined_df_1['INTIME'] = pd.to_datetime(combined_df_1['INTIME'])
combined_df_1['OUTTIME'] = pd.to_datetime(combined_df_1['OUTTIME'])
combined_df_1['INTIME'] < combined_df_1['TFLUID']

index_notinrange_1 = combined_df_1[(combined_df_1['INTIME'] < combined_df_1['TFLUID'])==False].index
combined_df_1.drop(index_notinrange_1, inplace=True)
index_notinrange_2 = combined_df_1[(combined_df_1['TFLUID'] < combined_df_1['OUTTIME'])==False].index
combined_df_1.drop(index_notinrange_2,inplace=True)


# Merge above df with waveform
combined_df_2 = pd.merge(combined_df_1,
                        emrdata,
                       on = 'ICUSTAY_ID')

# Check shapes 
combined_df_2.shape
emrdata.shape
merged_df_mice_imputed.shape


# Export to csv file
combined_df_2.to_csv('merged_file.csv')