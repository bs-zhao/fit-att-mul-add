import pickle
import numpy as np

model_names = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
    "aRACE_1",
    "aRACE_2",
    "aRACE_t",
    "aRACE_g",
][-2:]

real_max_tp = 28287
real_max_nf = 128

sx = '1'
gens = [f"gen{i}" for i in range(1, 6)]

for model_name in model_names:
    rts, nfs = [], []
    for gen in gens:
        with open(f"../../../outputs/krajbich2010/s1_gen{sx}/{model_name}/{gen}.pkl", 'rb') as f:
            raw_data = pickle.load(f)
        for e in raw_data:
            rts += list(e['rt'])
            nfs += list(e['nf'])
    max_len_rt = max(real_max_tp, int(np.percentile(rts, 99.75) / 0.001))
    max_len_nf = max(real_max_nf, int(np.percentile(nfs, 99.75)))
    print(model_name, max_len_rt, max_len_nf)
    with open(f"../../../outputs/krajbich2010/s2_max_len{sx}_rt_{model_name}.txt", 'w') as f:
        f.write(f"{max_len_rt}\n")
    with open(f"../../../outputs/krajbich2010/s2_max_len{sx}_nf_{model_name}.txt", 'w') as f:
        f.write(f"{max_len_nf}\n")
