import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_names = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
    "aRACE_1",
    "aRACE_2",
    "aRACE_t",
    "aRACE_g",
][2:4]

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

    plt.figure(figsize=(8, 5))
    plt.hist(rts, bins=60)
    plt.axvline(max_len_rt * 0.001, linestyle='--', label='max_len_rt')
    plt.xlabel('RT (s)')
    plt.ylabel('Count')
    plt.title(f'{model_name}: RT')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../../outputs/krajbich2010/s2_hist_rt_{model_name}.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(nfs, bins=60)
    plt.axvline(max_len_nf, linestyle='--', label='max_len_nf')
    plt.xlabel('Number of fixations')
    plt.ylabel('Count')
    plt.title(f'{model_name}: Fixation count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../../../outputs/krajbich2010/s2_hist_nf_{model_name}.png', dpi=150)
    plt.close()
