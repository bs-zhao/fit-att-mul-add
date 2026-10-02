import pickle,numpy as np
from tools.model_info import model_names
real_max_tp=15000;real_max_nf=128;sx='1';gens=[f'gen{i}' for i in range(1,6)]
for m in model_names:
 rts=[];nfs=[]
 for g in gens:
  with open(f'./outputs/s1_gen{sx}/{m}/{g}.pkl','rb') as f:raw=pickle.load(f)
  for e in raw:rts+=list(e['rt']);nfs+=list(e['nf'])
 mrt=max(real_max_tp,int(np.percentile(rts,99.75)/0.001));mnf=max(real_max_nf,int(np.percentile(nfs,99.75)))
 with open(f'outputs/s2_max_len{sx}_rt_{m}.txt','w') as f:f.write(f'{mrt}\n')
 with open(f'outputs/s2_max_len{sx}_nf_{m}.txt','w') as f:f.write(f'{mnf}\n')
