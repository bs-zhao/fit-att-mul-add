# fit-att-mul-add

Behavioral SBI/model-comparison pipeline for the eight gaze-bias models.

The analysis code follows the organization and interfaces of `Cognition-Decision-Modeling-Lab/adm-sbi/G19_v3/nn_rv1`, but all executable analysis code lives under `run/fit_nn/` rather than the repository root.

```text
fit-att-mul-add/
├── data/                         # ignored
├── outputs/                      # ignored
├── packages/
│   └── c_bifood/                 # installable local simulator package
│       ├── c_bifood.pyx
│       └── setup.py
└── run/
    └── fit_nn/
        ├── s1_gen1w.py
        ├── s2_getmaxRT.py
        ├── s3_fe1.py
        ├── s4_train_range_huber.py
        ├── s5_fe_real_data.py
        ├── s5_fe_real_data_maxT.py
        ├── s5_fe_test_data_maxT.py
        ├── tools/
        └── compare_regen/
```

`c_bifood` and `data2param-flow` are imported as installed packages. Install them first:

```bash
pip install -e packages/c_bifood
pip install git+https://github.com/Cognition-Decision-Modeling-Lab/data2param-flow.git
```

Run analysis scripts from the repository root, for example:

```bash
python run/fit_nn/s1_gen1w.py
python run/fit_nn/s3_fe1.py
python run/fit_nn/s4_train_range_huber.py
```

Input data are kept in `data/`; all simulations, feature files, checkpoints and model-comparison outputs are kept in `outputs/`.
