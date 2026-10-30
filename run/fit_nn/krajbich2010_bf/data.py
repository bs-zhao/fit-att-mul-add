"""Legacy model equations and 17 trial features, with explicit variable-set masks."""
from pathlib import Path
import importlib.util
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
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

def feature(values, choice, rt, pos, dur):
    pos, dur = np.asarray(pos, dtype=np.int64), np.asarray(dur, dtype=np.int64)
    if not len(pos) or len(pos) != len(dur) or np.any(dur < 0) or not np.isin(pos, [0,1]).all():
        raise ValueError('Invalid binary fixation sequence')
    if choice not in (0, 1) or not np.isfinite(rt) or rt <= 0:
        raise ValueError('Invalid choice/RT')
    if len(pos) >= 128 or np.ceil(dur.sum()/3) >= np.floor(28287/3):
        return None
    summary = funcs.get_summary(funcs.arr2tp(pos, dur), .001)
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

    def simulate(self, model, rng, stats, params=None, group=None):
        from c_bifood import sim_trial_aDDM, sim_trial_aRACE
        info = model_infos[model]
        params = funcs.make_params(info, rng) if params is None else params
        group = self.groups[int(rng.integers(len(self.groups)))][1] if group is None else group
        simulator = sim_trial_aDDM if model.startswith('aDDM') else sim_trial_aRACE
        rows = []
        for _, r in group.iterrows():
            result = simulator(params, np.ascontiguousarray([r.v0,r.v1], dtype=np.float64),
                               np.ascontiguousarray(r.arr_pos, dtype=np.int64).copy(),
                               np.ascontiguousarray(r.arr_du, dtype=np.int64).copy(),
                               extend_last=500000, repeat=0, dt=.001, rcd=1, max_rt=MAX_RT)
            (choice, rt, hit), (_, _, pos, dur), *_ = result
            stats['trials'] += 1
            if not hit:
                stats['nonhits'] += 1
                continue
            x = feature([r.v0,r.v1], int(choice), float(rt), pos, dur)
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
