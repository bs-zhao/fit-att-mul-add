import glob,os,sys,pandas as pd,torch
from data2param_flow import load_instance
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from tools.model_info import model_names
feature_d='trial';feature='trial';sx='1';dp=0.15;save_name='_'.join(model_names);dec=load_instance(f'../../../../outputs/food_equal/real_data/{feature_d}/dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/ParameterDecoder/parameter_decoder.pkl');dec.batch_size=64;rows=[]
for true_model in model_names:
 files=sorted(glob.glob(f'../../../../outputs/food_equal/real_data/{feature_d}/s3_fe{sx}_maxT/{true_model}/gen*.pkl'))
 if not files:continue
 prob=torch.softmax(dec.predict(files[-2:],post_dropout=True)[0],dim=1).cpu().numpy()
 for r in prob:rows.append({'true_model':true_model,**{m:r[i] for i,m in enumerate(model_names)}})
df=pd.DataFrame(rows);os.makedirs('../../../../outputs/food_equal/real_data/trial/s6_decode_trial_mr',exist_ok=True);df.to_csv(f'../../../../outputs/food_equal/real_data/trial/s6_decode_trial_mr/{save_name}.csv',index=False);print(df.groupby('true_model')[model_names].mean())
