import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import keras
import bayesflow as bf
from .data import ROOT, MODEL_NAMES, model_infos, FEATURE_NAMES, SCALE, Experiment

def positive(value):
    n=int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return n

def nonnegative(value):
    n=int(value)
    if n < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return n

def parser(description, model=False, classifier=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--mode', choices=['test','formal'], default='formal')
    p.add_argument('--input', type=Path, default=ROOT/'data/krajbich2010/trial_eye.csv')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/krajbich2010_bf')
    p.add_argument('--seed', type=nonnegative, default=20261030)
    if model:
        p.add_argument('--model', choices=MODEL_NAMES, required=True)
    if classifier:
        p.add_argument('--models', nargs='+', choices=MODEL_NAMES, metavar='MODEL',
                       help='Choose any 2-8 distinct models; omit for all eight')
    return p


def selected_models(args):
    """Return selected models in canonical order, independent of CLI order."""
    requested = getattr(args, 'models', None)
    if requested is None:
        return list(MODEL_NAMES)
    if not 2 <= len(requested) <= len(MODEL_NAMES):
        raise ValueError('--models must specify between 2 and 8 models')
    if len(set(requested)) != len(requested):
        raise ValueError('--models cannot contain duplicates')
    unknown = set(requested) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f'Unknown models: {sorted(unknown)}')
    return [name for name in MODEL_NAMES if name in requested]

def environment(seed):
    if version('bayesflow') != '2.1.0' or keras.backend.backend() != 'torch':
        raise RuntimeError('Requires bayesflow==2.1.0 and KERAS_BACKEND=torch')
    keras.utils.set_random_seed(seed)
    # c_bifood uses libc rand, independent of numpy / torch. Seed explicitly.
    import ctypes
    import sys
    ctypes.CDLL('msvcrt' if sys.platform=='win32' else None).srand(seed % (2**32))

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x:x.item() if isinstance(x,np.generic) else str(x)), encoding='utf8')

def directory(args, classifier=False):
    if not classifier:
        return args.output / args.model / args.mode
    names = selected_models(args)
    # Preserve all-eight paths; keep each proper subset in a separate tree.
    if names == MODEL_NAMES:
        return args.output / 'classifier' / args.mode
    return args.output / 'classifier' / '__'.join(names) / args.mode

def adapter(model=None):
    a = bf.Adapter().convert_dtype('float64','float32')
    if model:
        info = model_infos[model]
        for name, (lo,hi) in zip(info['free_pnames'],info['free_pranges']):
            a.constrain(name, lower=lo, upper=hi)
        a.concatenate(info['free_pnames'], into='inference_variables')
    else:
        a.rename('model_indices','inference_variables')
    return a.rename('trials','summary_variables').rename('n_trials','inference_conditions')

def workflow(simulator, model=None, model_names=None):
    kw = dict(simulator=simulator, adapter=adapter(model),
              summary_network=bf.networks.SetTransformer(summary_dim=32),
              standardize=['inference_variables'] if model else [],
              optimizer=keras.optimizers.Adam(learning_rate=5e-4,clipnorm=1.5))
    if model:
        return bf.BasicWorkflow(inference_network=bf.networks.CouplingFlow(transform='spline'), **kw)
    return bf.ModelComparisonWorkflow(
        model_names=MODEL_NAMES if model_names is None else model_names,
        scoring_rules='cross_entropy', **kw)

class AtomicBest(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path, self.best = Path(path), np.inf
    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get('val_loss',np.inf)
        if np.isfinite(loss) and loss < self.best:
            tmp = self.path.with_name(self.path.stem+'.tmp.keras')
            self.model.save(tmp)
            os.replace(tmp,self.path)
            self.best = float(loss)
            dump(self.path.with_suffix('.json'),dict(epoch=epoch+1,val_loss=self.best))

def training_args(p):
    for name in ['epochs','batches','batch-size','validation']:
        p.add_argument('--'+name,type=positive)
    p.add_argument('--overwrite',action='store_true')
    return p

def train(args, classifier=False):
    environment(args.seed)
    ex = Experiment(args.input)
    folder = directory(args,classifier)
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'best.keras').exists() and not args.overwrite:
        raise FileExistsError(f'{folder}/best.keras exists; use --overwrite for intentional retraining')
    if args.overwrite:
        for name in ['best.keras','best.json','final.keras']:
            (folder/name).unlink(missing_ok=True)
    names = selected_models(args) if classifier else None
    stats = {}
    def make(model, seed, only):
        s, stats[model] = ex.simulator(bf,model,seed,only)
        return s
    if classifier:
        simulator = bf.simulators.ModelComparisonSimulator([make(m,args.seed+1000*i,True) for i,m in enumerate(names)])
    else:
        simulator = make(args.model,args.seed,False)
    # Separate RNG streams for validation and training (C rand remains a single process stream).
    if classifier:
        validation_sim = bf.simulators.ModelComparisonSimulator([ex.simulator(bf,m,args.seed+100000+1000*i,True)[0] for i,m in enumerate(names)])
    else:
        validation_sim = ex.simulator(bf,args.model,args.seed+100000)[0]
    cfg = dict(epochs=args.epochs or (5 if args.mode=='test' else 5000),
               num_batches_per_epoch=args.batches or (2 if args.mode=='test' else 50),
               batch_size=args.batch_size or (8 if args.mode=='test' else 64))
    validation = validation_sim.sample(args.validation or (16 if args.mode=='test' else 2000))
    w = workflow(simulator,None if classifier else args.model,model_names=names)
    dump(folder/'metadata.json',dict(bayesflow=version('bayesflow'), seed=args.seed,
         model_names=names if classifier else MODEL_NAMES, model=None if classifier else args.model,
         parameter_names=None if classifier else model_infos[args.model]['free_pnames'],
         input_sha256=ex.sha256, max_trials=ex.max_trials, features=FEATURE_NAMES,
         feature_scale=SCALE.tolist(), config=cfg,
         model_prior=[1/len(names)]*len(names) if classifier else None))
    callbacks=[AtomicBest(folder/'best.keras'), keras.callbacks.CSVLogger(str(folder/'history.csv')),
               keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=8,min_lr=1e-6),
               keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,min_delta=1e-4,restore_best_weights=True)]
    w.fit_online(**cfg, validation_data=validation, callbacks=callbacks)
    w.estimator.save(folder/'final.keras')
    dump(folder/'simulation_stats.json',stats)

def load(args, classifier=False):
    environment(args.seed)
    folder=directory(args,classifier)
    meta=json.loads((folder/'metadata.json').read_text())
    if classifier and meta['model_names'] != selected_models(args):
        raise ValueError('Checkpoint model_names do not match --models selection')
    ex=Experiment(args.input)
    if ex.sha256 != meta['input_sha256']:
        raise ValueError('Input CSV differs from the training design. Use the original training CSV.')
    return keras.saving.load_model(folder/'best.keras'),ex,folder

def posterior(estimator, raw, names, draws, batch_size=25):
    samples=estimator.sample(conditions=raw,num_samples=draws,batch_size=batch_size)
    result=np.concatenate([np.asarray(samples[n]) for n in names],axis=-1)
    if result.shape != (len(raw['trials']),draws,len(names)):
        raise RuntimeError(f'Unexpected posterior shape: {result.shape}')
    return result
