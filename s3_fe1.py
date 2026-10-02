import os,pickle,numpy as np
from tqdm import tqdm
from tools.funcs import arr2tp,get_summary
from tools.model_info import model_names
dt=0.001;batch_size=500;sx='1';gens=[f'gen{i}' for i in range(1,51)];eye2=np.eye(2)
for m in model_names:
 src=f'./outputs/s1_gen{sx}/{m}/';dst=f'./outputs/s3_fe{sx}/{m}/';os.makedirs(dst,exist_ok=True)
 for g in gens:
  with open(f'{src}/{g}.pkl','rb') as f:raw=pickle.load(f)
  for ib in range(0,len(raw),batch_size):
   fe=[]
   for s in tqdm(raw[ib:ib+batch_size],leave=False):
    d={'trialinfo':np.asarray(s['vs'],dtype=float),'choice':eye2[np.asarray(s['choice'],dtype=int)],'rt':np.asarray(s['rt'],dtype=float),'fsmr':np.asarray([get_summary(arr2tp(p,du),dt) for p,du in zip(s['arr_pos'],s['arr_du'])],dtype=float),'params':s['params']}
    if len(d['rt'])>5:fe.append(d)
   with open(f'{dst}/{g}_batch{ib}.pkl','wb') as f:pickle.dump(fe,f,protocol=4)
