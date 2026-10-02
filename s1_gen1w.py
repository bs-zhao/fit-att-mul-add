import os,pickle,random,numpy as np
from tqdm import tqdm
from c_bifood import sim_trial_aDDM,sim_trial_aRACE
from tools.model_info import model_infos,model_names
from tools.funcs import load_food_data,make_params
dt=0.001;max_rt=14.5;for_test=0;n_round=500;names=[f'gen{i}' for i in range(1,51)]
for model_name in model_names:
 N_GEN=len(names);dir_save=f'./outputs/s1_gen1/{model_name}/'
 if for_test==1:names=['gen1'];N_GEN=1;dir_save=f'./outputs/s1_gen1_test/{model_name}/'
 os.makedirs(dir_save,exist_ok=True);df=load_food_data('data/trial_eye.csv');ids=list(df['subj'].unique());info=model_infos[model_name]
 for GEN in names:
  print(f'{model_name}: {GEN}/{N_GEN}');gen=[]
  for rd in tqdm(range(n_round)):
   subj=random.choice(ids);ds=df[df['subj']==subj];nums=np.array([ds.v0.values,ds.v1.values]);params=make_params(info);d={'subj':subj};ps=[];dus=[];nfs=[];cs=[];rts=[];vss=[]
   for i,(_,r) in enumerate(ds.iterrows()):
    v=np.ascontiguousarray(nums[:,i],dtype=np.float64);p=np.ascontiguousarray(np.asarray(r.arr_pos,dtype=np.int64));du=np.ascontiguousarray(np.asarray(r.arr_du,dtype=np.int64))
    if 'aDDM' in model_name:(c,rt,hit),sim,*_=sim_trial_aDDM(params,v,p,du,extend_last=500000,repeat=0,dt=dt,rcd=1,max_rt=max_rt)
    else:(c,rt,hit),sim,*_=sim_trial_aRACE(params,v,p,du,extend_last=500000,repeat=0,dt=dt,rcd=1,max_rt=max_rt)
    if hit==0:continue
    e,a,sp,sd=sim;ps.append(np.asarray(sp));dus.append(np.asarray(sd));nfs.append(len(sp));cs.append(int(c));rts.append(float(rt));vss.append(v)
   if len(cs)<=5:continue
   d.update(arr_pos=np.array(ps,dtype=object),arr_du=np.array(dus,dtype=object),nf=np.array(nfs),choice=np.array(cs),rt=np.array(rts),vs=np.array(vss),params=params);gen.append(d)
  with open(f'{dir_save}/{GEN}.pkl','wb') as f:pickle.dump(gen,f,protocol=4)
