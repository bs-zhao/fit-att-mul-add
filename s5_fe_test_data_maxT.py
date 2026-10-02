import os,pickle,numpy as np
from tools.funcs import arr2tp,get_summary
from tools.model_info import model_names
dt=0.001;eye2=np.eye(2)
for m in model_names:
 src=f'outputs/s1_gen1_test/{m}/gen1.pkl'
 if not os.path.exists(src):continue
 with open(src,'rb') as f:raw=pickle.load(f)
 fe=[{'subj':s['subj'],'choice':eye2[np.asarray(s['choice'],dtype=int)],'rt':np.asarray(s['rt'],dtype=float),'params':s['params'],'trialinfo':np.asarray(s['vs'],dtype=float),'fsmr':np.asarray([get_summary(arr2tp(p,du),dt) for p,du in zip(s['arr_pos'],s['arr_du'])],dtype=float)} for s in raw];out=f'outputs/s3_fe1_maxT/test/{m}/fe.pkl';os.makedirs(os.path.dirname(out),exist_ok=True)
 with open(out,'wb') as f:pickle.dump(fe,f,protocol=4)
