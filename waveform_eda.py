import os 
import json

path= 'C:/Users/dongliel/Downloads/waveform'
wffiles= []

for i in os.listdir(path):
	if os.path.isfile(os.path.join(path, i)) and i.startswith("wfdict"):
		wffiles.append(i)

MYDICT= {
		"num_sid": 0, 
		"num_wfrec": 0,
		"num_tfrdict": 0,
		"num_abpfollowup": 0, 
		"sum_abpfollowup": 0, 

}


for each_file in wffiles:
	with open(each_file, "r") as f:
		f_in= json.load(f)
	for each_sid in list(f_in.keys()):
		res_sid= f_in[each_sid]
		if len(res_sid) == 0:
			continue
		for each_wfrec in res_sid:
			if (each_wfrec["num_tfr_within_range"] == 0) or each_wfrec["except_flag"]== 1:
				continue
			tfrdict_wfrec= each_wfrec["tfr_dict"]
			for each_tfrdict in list(tfrdict_wfrec.keys()):
				unit_dict= tfrdict_wfrec[each_tfrdict]
				freq= unit_dict["fs"]
				signal_list= unit_dict["sig_name"]
				for each_signal in signal_list:
					if each_signal not in MYDICT:
						MYDICT[each_signal]= {
									"count": 1,  #count as on waveform record level
									"sum_length": unit_dict["extracted_length"] #sec
						}
					else:
						MYDICT[each_signal]["count"]+=1
						MYDICT[each_signal]["sum_length"]+= unit_dict["extracted_length"]

				abp_followup_len= len(unit_dict["abp_followup"])/freq
				MYDICT["num_abpfollowup"]+= (abp_followup_len >0)
				MYDICT["sum_abpfollowup"]+= abp_followup_len
				MYDICT["num_tfrdict"]+= 1
			MYDICT["num_wfrec"]+=1

		MYDICT["num_sid"]+=1

FN_OUT= "eda.txt"
with open(FN_OUT, "w") as file:
	file.write(json.dumps(MYDICT))

