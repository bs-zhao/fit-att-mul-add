# fit-att-mul-add

Behavioral SBI/model-comparison project designed to support multiple experiments/datasets.

Each experiment gets its own complete `nn_rv1`-style pipeline, its own model definitions, its own data folder, and its own output folder. Shared installable dependencies such as `c_bifood` and `data2param-flow` remain outside the experiment folders.

```text
fit-att-mul-add/
├── data/                         # ignored
│   ├── food_equal/
│   │   └── trial_eye.csv
│   ├── experiment_2/
│   └── experiment_3/
├── outputs/                      # ignored
│   ├── food_equal/
│   ├── experiment_2/
│   └── experiment_3/
├── packages/
│   └── c_bifood/
└── run/
    └── fit_nn/
        ├── food_equal/
        │   ├── s1_gen1w.py
        │   ├── s2_getmaxRT.py
        │   ├── s3_fe1.py
        │   ├── s4_train_range_huber.py
        │   ├── s5_fe_real_data.py
        │   ├── s5_fe_real_data_maxT.py
        │   ├── s5_fe_test_data_maxT.py
        │   ├── tools/
        │   └── compare_regen/
        ├── experiment_2/
        └── experiment_3/
```

The current experiment is `food_equal`. Its eight model definitions and preprocessing are experiment-specific and live entirely under `run/fit_nn/food_equal/`.

Install shared packages once:

```bash
pip install -e packages/c_bifood
pip install git+https://github.com/Cognition-Decision-Modeling-Lab/data2param-flow.git
```

Run the current experiment from its own directory:

```bash
cd run/fit_nn/food_equal
python s1_gen1w.py
python s3_fe1.py
python s4_train_range_huber.py
```

Current data location:

```text
data/food_equal/trial_eye.csv
```

All generated files for this experiment go under:

```text
outputs/food_equal/
```
