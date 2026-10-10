from data2param_flow import ParameterDecoder
from tools.model_info import model_infos

model_names = [
    "aDDM_1",
    "aDDM_2",
    "aDDM_t",
    "aDDM_g",
    "aRACE_1",
    "aRACE_2",
    "aRACE_t",
    "aRACE_g",
][-2:-1]

feature = 'trial'
sx = '1'
max_trial = 20000
dp = 0.15

for model_name in model_names:
    data_dir = f"../../../outputs/krajbich2010/s3_fe{sx}/{model_name}/"
    model_info = model_infos[model_name]
    param_names = model_info['free_pnames_ml']
    ranges = model_info['free_pranges_ml']

    my_decoder = ParameterDecoder(
        dir_save=f'../../../outputs/krajbich2010/dpsRH{sx}{"_dp"+str(dp) if dp>0 else ""}/{feature}/{model_name}/')
    my_decoder.prepare_datafile(
        data_dir,
        key_data_trial=['trialinfo', 'choice', 'rt', 'fsmr'],
        key_param='params',
        param_names_use=param_names,
        batch_size=32,
        val_split=0.06,
        seq_input_stratagy=1,
        max_trial=max_trial,
        ranges=ranges)
    my_decoder.prepare_model(post_dropout=dp)
    my_decoder.train(20000, 20)
    my_decoder.save_instance()
