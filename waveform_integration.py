import pandas as pd 
import numpy as np 
import os
import json

path= 'C:/Users/dongliel/Downloads/waveform'
wffiles= []
#for i in os.listdir(path):
for i in os.listdir(path):
	if os.path.isfile(os.path.join(path, i)) and i.startswith("wfdict"):
		wffiles.append(i)

DEF_COLS= ["SUBJECT_ID", "TFLUID", "EXTRACTED_LENGTH"]
df= pd.DataFrame(columns= DEF_COLS)
SIGNAL_INCLUDE= ["II", "PLETH", "V"]

for each_file in wffiles:
	with open(each_file, "r") as f:
		f_in= json.load(f)
	for each_sid in list(f_in.keys()):
		res_sid= f_in[each_sid]
		if len(res_sid) == 0:
			continue
		for each_wfrec in res_sid:
			if (each_wfrec["num_tfr_within_range"] == 0) or each_wfrec["except_flag"] == 1:
				continue
			tfrdict_wfrec= each_wfrec["tfr_dict"]
			for each_tfrdict in list(tfrdict_wfrec.keys()):
				#each_tfrdict == t-fluid timestamp
				unit_dict= tfrdict_wfrec[each_tfrdict]

				mydict= {"SUBJECT_ID": each_sid, 
						 "TFLUID": each_tfrdict,
						 "EXTRACTED_LENGTH": unit_dict["extracted_length"]
				}

				freq= unit_dict["fs"]
				extraction_result= unit_dict["extraction_result"]
				nicer_view= extraction_result.replace("\\", "").replace("\"", "").replace("[", "").replace("{","").replace("}", "").replace("]", "").split(",")
				# print (nicer_view)
				# print (len(nicer_view))
				for each_line in nicer_view:
					if each_line.startswith(("AVR", "II", "V")) and not each_line.startswith("III"):
		
						col_in, val_in= each_line.split(":")
						#print (col_in)
						#print (val_in)
						#print(each_line)
						if col_in not in df:
							df[col_in]= np.nan 
						if val_in == "null":
							val_in = np.nan
						mydict[col_in]= float(val_in)

				df.loc[each_sid]= pd.Series(mydict)

df.to_csv("integrated.csv")








