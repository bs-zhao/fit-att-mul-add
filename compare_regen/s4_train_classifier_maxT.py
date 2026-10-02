from data2param_flow import ParameterDecoder
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from tools.model_info import model_names,model_infos
feature_d='trial';feature='trial';for_test=0;sx='1';max_trial=20000;dp=0.15;save_name='_'.join(model_names);base='mr' if for_test else 'real_data';dirs=[f'../outputs/{base}/{feature_d}/s3_fe{sx}_maxT/{m}/' for m in model_names];pnames=model_infos[model_names[-1]]['free_pnames_ml'];dec=ParameterDecoder(decoder_type='classify',dir_save=f'../outputs/{base}/{feature_d}/dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/');dec.prepare_datafile(dirs,key_data_trial=['trialinfo','choice','rt','fsmr'],key_param='params',param_names_use=pnames,batch_size=32,val_split=0.06,seq_input_stratagy=1,max_trial=max_trial);dec.prepare_model(post_dropout=dp);dec.train(20000,20);dec.save_instance()
