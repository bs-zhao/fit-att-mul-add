import os
import sys

import pandas as pd
import torch
from data2param_flow import load_instance

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

feature_d = 'trial'
feature = 'trial'
sx = '1'
dp = 0.15
save_name = '_'.join(model_names)
output_root = '../../../../outputs/food_equal/compare_regen'

path_d2p = (
    f'{output_root}/real_data/{feature_d}/'
    f'dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/'
    'ParameterDecoder/parameter_decoder.pkl'
)
path_data = f'../../../../outputs/food_equal/s3_fe{sx}_maxT/real_data/fe.pkl'

dec = load_instance(path_d2p)
dec.batch_size = 64
prob = torch.softmax(
    dec.predict([path_data], post_dropout=True)[0], dim=1
).cpu().numpy()

df = pd.DataFrame(prob, columns=model_names)
file_save = f'{output_root}/real_data/{feature_d}/s6_decode_{feature}/{save_name}.csv'
os.makedirs(os.path.dirname(file_save), exist_ok=True)
df.to_csv(file_save, index=False)
print(df.mean().sort_values(ascending=False))
print(f'saved {file_save}')
