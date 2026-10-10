from data2param import ParameterDecoder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools.model_info import model_infos

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]

model_names = [
    "aRACE_t",
    "aRACE_g",
]

feature_d = 'trial'
feature = 'trial'
for_test = 0
sx = '1'
max_trial = 20000
dp = 0.15
save_name = '_'.join(model_names)
base = 'mr' if for_test else 'real_data'
output_root = '../../../../outputs/krajbich2010/compare_regen'

dirs = [
    f'{output_root}/{base}/{feature_d}/s3_fe{sx}_maxT/{m}/'
    for m in model_names
]
pnames = model_infos[model_names[-1]]['free_pnames']

dec = ParameterDecoder(
    decoder_type='classify',
    dir_save=(
        f'{output_root}/{base}/{feature_d}/'
        f'dps_{feature_d}{sx}_dp{dp}_maxT/{feature}/{save_name}/'
    ),
)
dec.prepare_datafile(
    dirs,
    key_data_trial=['trialinfo', 'choice', 'rt', 'fsmr'],
    key_param='params',
    param_names_use=pnames,
    batch_size=32,
    val_split=0.06,
    seq_input_stratagy=1,
    max_trial=max_trial,
)
dec.prepare_model(post_dropout=dp)
dec.train(20000, 20)
dec.save_instance()
