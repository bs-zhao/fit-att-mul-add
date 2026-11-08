"""Legacy model equations and 17 trial features, with explicit variable-set masks."""
from pathlib import Path
import importlib.util
import hashlib
from copy import deepcopy
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
LEGACY = HERE.parent / 'food_equal'

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

funcs = _module('food_legacy_funcs', LEGACY / 'tools/funcs.py')
# Keep the original food_equal model_info.py untouched. The BF project
# operates on an independent copy with WIDE attention priors by default.
model_infos = deepcopy(_module('food_legacy_models', LEGACY / 'tools/model_info.py').model_infos)
MODEL_NAMES = list(model_infos)
ATTENTION_PRIORS = {"theta": (0.0, 1.0), "gamma": (0.0, 5.0)}
PRIOR_PROFILE = "wide_theta_0_1_gamma_0_5"

for _name, _info in model_infos.items():
    if not _name.endswith(("_t", "_g")):
        continue
    _parameter = "theta" if _name.endswith("_t") else "gamma"
    for _params_key, _ranges_key in (
        ("free_pnames", "free_pranges"),
        ("free_pnames_ml", "free_pranges_ml"),
    ):
        _index = _info[_params_key].index(_parameter)
        _bounds = list(_info[_ranges_key])
        _bounds[_index] = ATTENTION_PRIORS[_parameter]
        _info[_ranges_key] = _bounds

def prior_spec(names):
    """Exact generating priors for S1/S4 checkpoint compatibility checks."""
    return {
        "profile": PRIOR_PROFILE,
        "free_priors": {
            name: {parameter: [float(lo), float(hi)]
                   for parameter, (lo, hi) in zip(
                       model_infos[name]["free_pnames"],
                       model_infos[name]["free_pranges"])}
            for name in names
        },
    }
FEATURE_NAMES = ['v0', 'v1', 'choice0', 'choice1', 'rt', 'ttft0', 'ttft1',
                 'relft0', 'relft1', 'numf', 'numf0', 'numf1', 'fstf0', 'fstf1',
                 'lstf0', 'lstf1', 'switches']
maxT = _module('food_equal_legacy_maxT', LEGACY / 'tools/maxT.py')
MAX_LEN_RT = int(maxT.MAX_LEN_RT)  # 15,000 ms, as in food_equal/s5_fe_real_data_maxT.py
MAX_LEN_NF = int(maxT.MAX_LEN_NF)  # 128 fixation segments
MAX_RT = 14.5  # food_equal/s1_gen1w.py; unchanged simulator cutoff
SCALE = np.array([10,10,1,1,MAX_LEN_RT/1000,MAX_LEN_RT/1000,MAX_LEN_RT/1000,
                  1,1,MAX_LEN_NF,MAX_LEN_NF,MAX_LEN_NF,1,1,1,1,MAX_LEN_NF], np.float32)

def fast_fixation_summary(pos, dur):
    """Exactly reproduce get_summary(arr2tp(pos,dur), .001) without ms expansion.

    Merge adjacent equal fixation codes and ignore zero-duration segments,
    just as itertools.groupby on the expanded millisecond eye trace does.
    """
    # Match food_equal/tools/funcs.py:get_summary(arr2tp(pos,dur),.001):
    # that implementation expands all fixations, then excludes every gaze
    # code other than 0 or 1 before computing runs and dwell times.
    # Filtering invalid codes AFTER simulation preserves c_bifood dynamics.
    valid = (dur > 0) & ((pos == 0) | (pos == 1))
    if not np.any(valid):
        return [0.0] * 12
    p = pos[valid]
    d = dur[valid]
    n0 = int(d[p == 0].sum())
    n1 = int(d[p == 1].sum())
    total_ms = n0 + n1
    rt = max(total_ms * .001, 1e-9)
    starts = np.empty(len(p), dtype=bool)
    starts[0] = True
    starts[1:] = p[1:] != p[:-1]
    runs = p[starts]
    numf = len(runs)
    first, last = int(p[0]), int(p[-1])
    return [
        n0 * .001, n1 * .001,
        (n0 * .001) / rt, (n1 * .001) / rt,
        numf, int(np.count_nonzero(runs == 0)),
        int(np.count_nonzero(runs == 1)),
        float(first == 0), float(first == 1),
        float(last == 0), float(last == 1),
        max(numf - 1, 0),
    ]


def reconstruct_fixations(pos, dur, rt, ndt=0.0):
    """Recover the rcd=1 truncated gaze sequence from rcd=0 RT, at dt=.001.

    c_bifood extends ONLY the final fixation by 500000 ms. Its simulator
    terminates at an integer 1-ms step. The same RNG draws occur with either
    recording mode; only unused accumulator trajectories are omitted.
    """
    steps = int(np.rint((float(rt) - float(ndt)) / .001))
    if steps < 1 or not np.isfinite(rt):
        raise RuntimeError(f"Invalid simulated decision time {rt}")
    boundaries = np.cumsum(dur, dtype=np.int64)
    boundaries[-1] += 500000
    index = int(np.searchsorted(boundaries, steps, side="left"))
    if index >= len(pos):
        raise RuntimeError(f"Decision step {steps} exceeds extended gaze duration")
    out_dur = np.array(dur[:index + 1], copy=True)
    out_dur[-1] = steps - (int(boundaries[index-1]) if index else 0)
    if out_dur[-1] <= 0:
        raise RuntimeError(f"Invalid last fixation duration: {out_dur[-1]}")
    return pos[:index + 1], out_dur


def feature(values, choice, rt, pos, dur, legacy=False):
    pos, dur = np.asarray(pos, dtype=np.int64), np.asarray(dur, dtype=np.int64)
    if not len(pos) or len(pos) != len(dur) or np.any(dur < 0):
        raise ValueError('Invalid fixation sequence lengths/durations')
    # The legacy food_equal simulator handles any non-1 gaze position in
    # its existing else branch. The legacy get_summary ignores non-0/1
    # positions. Preserve BOTH behaviors; do not recode gaze pre-simulation.
    if choice not in (0, 1) or not np.isfinite(rt) or rt <= 0:
        raise ValueError('Invalid choice/RT')
    if len(pos) >= MAX_LEN_NF or np.ceil(dur.sum()/3) >= np.floor(MAX_LEN_RT/3):
        return None
    summary = (funcs.get_summary(funcs.arr2tp(pos, dur), .001)
               if legacy else fast_fixation_summary(pos, dur))
    x = np.r_[values, np.eye(2)[choice], rt, summary].astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feature')
    return x / SCALE

def pack(rows, max_trials):
    if len(rows) <= 5 or len(rows) > max_trials:
        raise ValueError(f'Invalid retained trial count {len(rows)} (maximum {max_trials})')
    x = np.zeros((max_trials, 17), np.float32)
    x[:len(rows)] = rows
    return {'trials': x, 'summary_mask': np.arange(max_trials) < len(rows),
            'n_trials': np.array([np.log1p(len(rows))], np.float32)}

class Experiment:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.df = funcs.load_food_data(self.path)
        self.groups = list(self.df.groupby('subj', sort=False))
        # Preconvert the reusable empirical value/fixation schedules once.
        # Cython internally copies positions/durations before mutating them.
        self._designs = [
            [(np.ascontiguousarray([r.v0, r.v1], dtype=np.float64),
              np.ascontiguousarray(r.arr_pos, dtype=np.int64),
              np.ascontiguousarray(r.arr_du, dtype=np.int64))
             for _, r in g.iterrows()]
            for _, g in self.groups
        ]
        if not self.groups:
            raise ValueError('No valid participants')
        self.max_trials = max(len(g) for _, g in self.groups)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def real(self):
        """Encode genuine food_equal responses, auditing excluded trials.

        The legacy loader retains trials with a valid fixation-array shape
        even when response is not binary or summed fixation RT is zero.
        Such trials cannot be represented by the binary-choice likelihood.
        Do not invent a response or RT, and do not modify simulations or
        checkpoints: omit only the invalid *observed* trial, with its reason.
        """
        ids, records, audit = [], [], []
        exclusions = []
        for sid, group in self.groups:
            rows = []
            reasons = {
                'invalid_choice': 0,
                'invalid_rt': 0,
                'invalid_fixation': 0,
                'invalid_values': 0,
                'maxT_excluded': 0,
            }
            for row_index, r in group.iterrows():
                values = np.asarray([r.v0, r.v1], dtype=np.float64)
                pos = np.asarray(r.arr_pos, dtype=np.int64)
                dur = np.asarray(r.arr_du, dtype=np.int64)
                choice = r.choice
                rt = float(r.rt)
                if not np.isfinite(choice) or choice not in (0, 1):
                    reason = 'invalid_choice'
                elif not np.isfinite(rt) or rt <= 0:
                    reason = 'invalid_rt'
                elif (not len(pos) or len(pos) != len(dur)
                      or np.any(dur < 0)):
                    reason = 'invalid_fixation'
                elif not np.isfinite(values).all():
                    reason = 'invalid_values'
                else:
                    x = feature(values, int(choice), rt, pos, dur)
                    if x is None:
                        reason = 'maxT_excluded'
                    else:
                        rows.append(x)
                        continue
                reasons[reason] += 1
                exclusions.append({
                    'subj': sid,
                    'row_index': row_index,
                    'reason': reason,
                    'raw_choice': choice,
                    'raw_rt_s': rt,
                    'n_fixations': len(pos),
                    'total_fixation_ms': int(dur.sum()),
                })
            audit.append({
                'subj': sid,
                'input_trials': len(group),
                'retained_trials': len(rows),
                'excluded_trials': len(group)-len(rows),
                **reasons,
                'included': len(rows) > 5,
            })
            if len(rows) > 5:
                ids.append(sid)
                records.append(pack(rows, self.max_trials))
        self.real_exclusions = exclusions
        if not records:
            raise ValueError('No participant has >5 retained valid trials; inspect input CSV and trial exclusions')
        return ids, {k:np.stack([r[k] for r in records]) for k in records[0]}, audit

    def simulate(self, model, rng, stats, params=None, group=None, legacy=False):
        """Simulate unchanged 1-ms dynamics without unnecessary trace recording.

        legacy=True reproduces the original path for exact parity testing.
        Production uses cached empirical designs, rcd=0, and direct summary.
        """
        from c_bifood import sim_trial_aDDM, sim_trial_aRACE
        info = model_infos[model]
        params = funcs.make_params(info, rng) if params is None else params
        if group is None:
            design_idx = int(rng.integers(len(self.groups)))
            if legacy:
                group = self.groups[design_idx][1]
            else:
                design = self._designs[design_idx]
        if legacy:
            if group is None:
                raise RuntimeError("Legacy simulation requires an empirical group")
            design = [
                (np.ascontiguousarray([r.v0, r.v1], dtype=np.float64),
                 np.ascontiguousarray(r.arr_pos, dtype=np.int64).copy(),
                 np.ascontiguousarray(r.arr_du, dtype=np.int64).copy())
                for _, r in group.iterrows()
            ]
        elif group is not None:
            design = [
                (np.ascontiguousarray([r.v0, r.v1], dtype=np.float64),
                 np.ascontiguousarray(r.arr_pos, dtype=np.int64),
                 np.ascontiguousarray(r.arr_du, dtype=np.int64))
                for _, r in group.iterrows()
            ]
        simulator = sim_trial_aDDM if model.startswith('aDDM') else sim_trial_aRACE
        rows = []
        for values, orig_pos, orig_dur in design:
            if legacy:
                result = simulator(params, values, orig_pos.copy(), orig_dur.copy(),
                                   extend_last=500000, repeat=0, dt=.001,
                                   rcd=1, max_rt=MAX_RT)
                (choice, rt, hit), (_, _, pos, dur), *_ = result
            else:
                (choice, rt, hit), _ = simulator(
                    params, values, orig_pos, orig_dur,
                    extend_last=500000, repeat=0, dt=.001,
                    rcd=0, max_rt=MAX_RT)
            stats['trials'] += 1
            if not hit:
                stats['nonhits'] += 1
                continue
            if not legacy:
                pos, dur = reconstruct_fixations(
                    orig_pos, orig_dur, rt, ndt=params.get('ndt', 0.0))
            x = feature(values, int(choice), float(rt), pos, dur, legacy=legacy)
            if x is None:
                stats['maxT_excluded'] += 1
            else:
                rows.append(x)
        return params, rows

    def simulator(self, bf, model, seed, model_only=False):
        rng = np.random.default_rng(seed)
        stats = dict(trials=0, nonhits=0, maxT_excluded=0, rejected_subjects=0, accepted_subjects=0)
        def one():
            for _ in range(100):
                params, rows = self.simulate(model, rng, stats)
                if len(rows)>5:
                    stats['accepted_subjects'] += 1
                    record = pack(rows, self.max_trials)
                    if not model_only:
                        record.update({p:np.array([params[p]], np.float32) for p in model_infos[model]['free_pnames']})
                    return record
                stats['rejected_subjects'] += 1
            raise RuntimeError(f'{model}: 100 rejected subjects; inspect priors/design/nonhits: {stats}')
        return bf.make_simulator(one), stats
