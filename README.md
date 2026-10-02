# fit-att-mul-add

Behavioral SBI/model-comparison pipeline for the eight gaze-bias models.

This repository deliberately mirrors the file layout and script interfaces of `Cognition-Decision-Modeling-Lab/adm-sbi/G19_v3/nn_rv1`. Only the parts required for the present binary food-choice dataset and the eight submitted models are changed.

- input data: `data/trial_eye.csv`
- all generated data, checkpoints, predictions and comparison outputs: `outputs/`
- parameter decoder: `data2param-flow` (`from data2param_flow import ParameterDecoder`)
- simulator: original binary-food aDDM/aRACE equations in `c_bifood.pyx`

Run order follows `nn_rv1`: `s1_gen1w.py` -> `s2_getmaxRT.py` -> `s3_fe1.py` -> `s4_train_range_huber.py` -> `s5_fe_real_data*.py`, then `compare_regen/`.

Before running, install `data2param-flow` and compile the simulator:

```bash
pip install git+https://github.com/Cognition-Decision-Modeling-Lab/data2param-flow.git
python setup.py build_ext --inplace
```
