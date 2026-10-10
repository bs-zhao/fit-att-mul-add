import os
import sys

import numpy as np
import pandas as pd
from data2param import load_instance

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]


feature_d = 'trial'
feature = 'trial'
sx = '1'
max_trial = 20000
dp = 0.15
save_name = '_'.join(model_names)
output_root = '../../../../outputs/krajbich2010/compare_regen'

path_d2p = (
    f'{output_root}/mr/{feature_d}/'
    f'dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/'
    'ParameterDecoder/parameter_decoder.pkl'
)

dec = load_instance(path_d2p)
if not hasattr(dec, 'max_trial'):
    dec.max_trial = 2000
dec.batch_size = 1

for true_model in model_names:
    path_data = (
        f'../../../../outputs/krajbich2010/s3_fe{sx}_maxT/test/'
        f'{true_model}/fe.pkl'
    )
    out = dec.predict([path_data], post_dropout=True)[0].cpu().numpy()

    print(f'\ntrue model: {true_model}')
    for i in range(out.shape[1]):
        print(model_names[i], np.mean(out[:, i]))

    file_save = (
        f'{output_root}/mr/{feature_d}/s6_decode_{feature}/'
        f'{true_model}/{save_name}.csv'
    )
    os.makedirs(os.path.dirname(file_save), exist_ok=True)
    pd.DataFrame(out, columns=model_names).to_csv(file_save, index=False)
    print(f'saved {file_save}')
