import os,pickle,numpy as np
from tools.funcs import load_food_data,arr2tp,get_summary
sx='1';dt=0.001;eye2=np.eye(2);out=f'outputs/s3_fe{sx}_maxT/real_data/fe.pkl';os.makedirs(os.path.dirname(out),exist_ok=True);df=load_food_data('data/trial_eye.csv');df.to_csv(f'outputs/s3_fe{sx}_maxT/real_data/df.csv',index=False);fe=[]
for subj in df['subj'].unique():
 ds=df[df['subj']==subj];d={'subj':subj,'choice':eye2[np.asarray(ds.choice,dtype=int)],'rt':np.asarray(ds.rt,dtype=float),'trialinfo':np.array([ds.v0.values,ds.v1.values]).T,'fsmr':np.asarray([get_summary(arr2tp(np.asarray(r.arr_pos,dtype=np.int64),np.asarray(r.arr_du,dtype=np.int64)),dt) for _,r in ds.iterrows()],dtype=float),'params':None};fe.append(d)
with open(out,'wb') as f:pickle.dump(fe,f,protocol=4)
