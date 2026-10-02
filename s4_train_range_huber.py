from data2param_flow import ParameterDecoder
from tools.model_info import model_infos,model_names
feature='trial';sx='1';max_trial=20000;dp=0.15
for m in model_names:
 info=model_infos[m];dec=ParameterDecoder(dir_save=f'./outputs/dpsRH{sx}{"_dp"+str(dp) if dp>0 else ""}/{feature}/{m}/');dec.prepare_datafile(f'outputs/s3_fe{sx}/{m}/',key_data_trial=['trialinfo','choice','rt','fsmr'],key_param='params',param_names_use=info['free_pnames_ml'],batch_size=32,val_split=0.06,seq_input_stratagy=1,max_trial=max_trial,ranges=info['free_pranges_ml']);dec.prepare_model(post_dropout=dp);dec.train(20000,20);dec.save_instance()
