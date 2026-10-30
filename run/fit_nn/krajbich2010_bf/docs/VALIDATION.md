# Validation performed 2026-10-30

Environment: Python 3.12, BayesFlow 2.1.0, Keras torch backend, CPU PyTorch. The original `packages/c_bifood/c_bifood.pyx` was compiled with GCC/G++. No simulator equation was changed.

## Automated suite

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s run/fit_nn/krajbich2010_bf/tests -v`

Result: **5 tests passed** (40.2 seconds). Tests cover:

1. Numeric equality of the 17-feature representation with the legacy extraction, maxT exclusion and invalid-choice rejection.
2. Variable-length masks, zero padding, and Scheme A observed RT.
3. All eight native simulators, bounded adapters, parameter order/shapes, and protection against mutation of empirical fixation durations.
4. SetTransformer output invariance to changes in padded values and permutation of valid/padded trials.
5. One training batch, full `.keras` save/load, and bounded finite posterior samples for **all eight** models; eight-class training/save/load, probability normalization, and BayesFlow log-BF consistency with posterior odds under a uniform training prior.

## Independent command-line checks

All six entrypoints were run on a generated two-participant test CSV (12 and 15 trials), using the native simulator. Parameter stages used `aDDM_1`; the classifier included all eight models.

- s1: 1 epoch / 1 batch; best and final models saved.
- s2: 8 fresh datasets, 20 posterior draws; recovery/SBC/coverage/correlation CSVs and figures produced.
- s3: 2 participants, 20 posterior draws, 3 PPC draws; posterior shape `(2,20,2)`, estimates and PPC output produced.
- s4: 1 epoch / 1 mixed batch, 8 classes; checkpoint saved.
- s5: 4 fresh datasets per model; all confusion/calibration metrics and figures produced.
- s6: 2 participants; PMPs and 112 ordered pairwise log-BF rows produced. PMPs sum to one; pairwise log BFs are antisymmetric.

Python compilation and shell syntax checks passed. The upstream Keras/PyTorch stack emits NumPy array-conversion deprecation warnings; these did not prevent training, serialization, sampling or diagnostics.

These checks establish implementation/API execution, not fitted-model quality. The real Krajbich participant CSV is not present in the tracked repository; no full real-data training or scientific recovery claim is made. GPU/Slurm execution has not been tested from this CPU workspace.

## Folder reorganization check

After moving shared modules into `_shared/` and updating all six entrypoint imports, the same five-test suite passed again. The top-level file inventory was checked to contain exactly the six `s1`–`s6` Python scripts; requirements, documentation and tests live in subfolders. Shell wrappers were removed in favor of direct `sbatch --wrap` submissions.
