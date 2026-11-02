"""Legacy model equations and 17 trial features, with explicit variable-set masks."""
from pathlib import Path
import importlib.util
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
LEGACY = HERE.parent / 'krajbich2010'

def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

funcs = _module('kraj_legacy_funcs', LEGACY / 'tools/funcs.py')
model_infos = _module('kraj_legacy_models', LEGACY / 'tools/model_info.py').model_infos
MODEL_NAMES = list(model_infos)
FEATURE_NAMES = ['v0', 'v1', 'choice0', 'choice1', 'rt', 'ttft0', 'ttft1',
                 'relft0', 'relft1', 'numf', 'numf0', 'numf1', 'fstf0', 'fstf1',
                 'lstf0', 'lstf1', 'switches']
SCALE = np.array([10,10,1,1,28.287,28.287,28.287,1,1,128,128,128,1,1,1,1,128], np.float32)
MAX_RT = 28.287

def fast_fixation_summary(pos, dur):
    """Exactly reproduce get_summary(arr2tp(pos,dur), .001) without ms expansion.

    Merge adjacent equal fixation codes and ignore zero-duration segments,
    just as itertools.groupby on the expanded millisecond eye trace does.
    """
    positive = dur > 0
    if not np.any(positive):
        return [0.0] * 12
    p = pos[positive]
    d = dur[positive]
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
    if not len(pos) or len(pos) != len(dur) or np.any(dur < 0) or not np.isin(pos, [0,1]).all():
        raise ValueError('Invalid binary fixation sequence')
    if choice not in (0, 1) or not np.isfinite(rt) or rt <= 0:
        raise ValueError('Invalid choice/RT')
    if len(pos) >= 128 or np.ceil(dur.sum()/3) >= np.floor(28287/3):
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
        ids, records, audit = [], [], []
        for sid, group in self.groups:
            rows = []
            for _, r in group.iterrows():
                x = feature([r.v0,r.v1], int(r.choice), float(r.rt), r.arr_pos, r.arr_du)
                if x is not None:
                    rows.append(x)
            audit.append({'subj': sid, 'input_trials':len(group), 'retained_trials':len(rows), 'included':len(rows)>5})
            if len(rows)>5:
                ids.append(sid)
                records.append(pack(rows, self.max_trials))
        if not records:
            raise ValueError('No participant has >5 retained trials')
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
