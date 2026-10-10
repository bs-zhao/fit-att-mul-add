from pathlib import Path as _Path
import os as _os
_SCRIPT_DIR = _Path(__file__).resolve().parent
_os.chdir(_SCRIPT_DIR)
from data2param import ParameterDecoder
import os
import sys

sys.path.append(str(_SCRIPT_DIR.parent))
from tools.model_info import model_infos

model_names = [
    "aDDM_1", "aDDM_2", "aDDM_t", "aDDM_g",
    "aRACE_1", "aRACE_2", "aRACE_t", "aRACE_g",
]


feature_d = 'trial'
feature = 'trial'
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--for-test', action='store_true')
ap.add_argument('--epochs', type=int, default=20000)
ap.add_argument('--patience', type=int, default=20)
args = ap.parse_args()
for_test = int(args.for_test)
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
pnames = ['d', 'a']  # common to all 8; classification labels come from directory

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
dec.train(args.epochs, args.patience)
dec.save_instance()
