import os,sys,pandas as pd,torch
from data2param_flow import load_instance
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

model_names = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
    "aRACE_1",
    "aRACE_2",
    "aRACE_t",
    "aRACE_g",
]

feature_d='trial';feature='trial';sx='1';dp=0.15;save_name='_'.join(model_names);dec=load_instance(f'../../../../outputs/food_equal/real_data/{feature_d}/dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/ParameterDecoder/parameter_decoder.pkl');dec.batch_size=64;prob=torch.softmax(dec.predict([f'../../../../outputs/food_equal/s3_fe{sx}_maxT/real_data/fe.pkl'],post_dropout=True)[0],dim=1).cpu().numpy();df=pd.DataFrame(prob,columns=model_names);os.makedirs('../../../../outputs/food_equal/real_data/trial/s6_decode_trial',exist_ok=True);df.to_csv(f'../../../../outputs/food_equal/real_data/trial/s6_decode_trial/{save_name}.csv',index=False);print(df.mean().sort_values(ascending=False))
