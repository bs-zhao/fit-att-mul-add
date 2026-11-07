"""Cross-architecture model comparison on four independent simulated mechanisms.

Generate held-out participants from DDM-Mul, DDM-Add, RACE-Mul, RACE-Add,
using the CURRENT wide priors (theta ~ U[0,1], gamma ~ U[0,5]).
Submit the SAME simulated datasets to frozen four-DDM and four-RACE S4
classifiers. Save full class posterior, all six pairwise log BFs, subject
metrics and figures, plus a combined 4-generators x 2-classifiers report.

No retraining; requires both completed wide-prior best.keras checkpoints.
From run/fit_nn/food_equal_bf:
  python -u experiments/cross_architecture_wide_prior.py --mode formal --n 1000
"""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import ctypes
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

GENERATORS = ('aDDM_t', 'aDDM_g', 'aRACE_t', 'aRACE_g')
FAMILIES = ('ddm', 'race')
TYPES = ('NoAtt', 'AttOnly', 'Mul', 'Add')
PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
LN10 = float(np.log(10))


def mechanism(generator):
    if generator not in GENERATORS:
        raise ValueError(f'Unexpected generator: {generator}')
    return 'Mul' if generator.endswith('_t') else 'Add'


def family_name(generator):
    return 'DDM' if generator.startswith('aDDM') else 'RACE'


def labels(generator):
    return f'{family_name(generator)}-{mechanism(generator)}'


def saved_figure(fig, out):
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def classifier_output(estimator, raw):
    """Get stable log-BFs and posterior model probabilities for four models."""
    result = estimator.estimate(conditions=raw, model_prior=np.full(4, .25))
    p = np.asarray(result['model_probs'], dtype=float)
    logev = np.asarray(result['log_bayes_factors'], dtype=float)
    n = len(raw['trials'])
    if p.shape != (n, 4) or logev.shape != (n, 4):
        raise ValueError(f'Wrong classifier output shape: {p.shape}, {logev.shape}')
    if not np.isfinite(p).all() or not np.isfinite(logev).all():
        raise ValueError('Nonfinite classification output')
    if (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-4):
        raise ValueError('Invalid posterior model probabilities')
    centered = logev - np.max(logev, axis=1, keepdims=True)
    q = np.exp(centered)
    q /= q.sum(1, keepdims=True)
    if not np.allclose(p, q, atol=1e-3):
        raise ValueError('log_bayes_factors inconsistent with model_probs')
    return p, logev


def evidence_counts(z):
    z = np.asarray(z, dtype=float)
    return {
        'strong_Add': int((z < -LN10).sum()),
        'weak_Add': int(((z >= -LN10) & (z < 0)).sum()),
        'weak_Mul': int(((z >= 0) & (z <= LN10)).sum()),
        'strong_Mul': int((z > LN10).sum()),
    }


def describe(frame, prefix, generator):
    z = frame[f'{prefix}_logBF_Mul_Add'].to_numpy(float)
    p = frame[[f'{prefix}_p_{name}' for name in TYPES]].to_numpy(float)
    chosen = frame[f'{prefix}_best_model'].to_numpy(str)
    true_mech = mechanism(generator)
    right = z if true_mech == 'Mul' else -z
    family_prefix = 'aDDM' if prefix == 'DDM' else 'aRACE'
    correct_four_model = family_prefix + ('_t' if true_mech == 'Mul' else '_g')
    counts = {name: int(np.sum(chosen == family_prefix + suffix))
              for name, suffix in zip(TYPES, ('_1', '_2', '_t', '_g'))}
    return {
        'n': len(z),
        'generator': generator,
        'classifier': prefix,
        'mean_logBF_Mul_Add': float(z.mean()),
        'median_logBF_Mul_Add': float(np.median(z)),
        'mean_logBF_correct_vs_wrong_mechanism': float(right.mean()),
        'correct_mechanism_rate': float(np.mean(right > 0)),
        'strong_correct_mechanism_rate': float(np.mean(right > LN10)),
        'strong_wrong_mechanism_rate': float(np.mean(right < -LN10)),
        'mechanism_named_model_win_rate': float(np.mean(chosen == correct_four_model)),
        'generator_family_is_classifier_family': bool(family_name(generator) == prefix),
        'winner_counts': counts,
        'mean_model_probabilities': {label: float(p[:, k].mean())
                                     for k, label in enumerate(TYPES)},
        'evidence_counts': evidence_counts(z),
    }


def paired_summary(frame, generator, bootstrap, seed):
    ddm = frame['DDM_logBF_Mul_Add'].to_numpy(float)
    race = frame['RACE_logBF_Mul_Add'].to_numpy(float)
    delta = ddm - race
    signed = delta if mechanism(generator) == 'Mul' else -delta
    n = len(delta)
    if n < 2:
        raise ValueError('Require at least 2 simulated participants')
    test = stats.ttest_rel(ddm, race)
    ci = stats.t.interval(.95, n-1, loc=delta.mean(), scale=stats.sem(delta))
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap, dtype=float)
    for j in range(bootstrap):
        boot[j] = delta[rng.integers(n, size=n)].mean()
    return {
        'mean_DDM_minus_RACE_logBF': float(delta.mean()),
        'mean_DDM_minus_RACE_correct_mechanism_logBF': float(signed.mean()),
        'paired_t': float(test.statistic),
        'paired_t_p_two_sided': float(test.pvalue),
        'paired_ci95': [float(v) for v in ci],
        'paired_bootstrap_ci95': [float(v) for v in np.quantile(boot, [.025, .975])],
        'n_DDM_logBF_higher': int((delta > 0).sum()),
        'n_RACE_logBF_higher': int((delta < 0).sum()),
        'n_DDM_correct_evidence_higher': int((signed > 0).sum()),
        'n_RACE_correct_evidence_higher': int((signed < 0).sum()),
        'bootstrap_samples': bootstrap,
    }


def pairwise_table(frame, generator):
    rows = []
    for prefix in ('DDM', 'RACE'):
        for i, j in PAIR_INDICES:
            col = f'{prefix}_logBF_{TYPES[i]}_{TYPES[j]}'
            z = frame[col].to_numpy(float)
            rows.append(dict(generator=generator, classifier=prefix,
                             numerator=TYPES[i], denominator=TYPES[j],
                             n=len(z), mean_logBF=float(z.mean()),
                             median_logBF=float(np.median(z)),
                             numerator_win_rate=float(np.mean(z > 0)),
                             denominator_win_rate=float(np.mean(z < 0)),
                             strong_numerator=int((z > LN10).sum()),
                             strong_denominator=int((z < -LN10).sum())))
    return pd.DataFrame(rows)


def individual_plots(frame, generator, out):
    out.mkdir(parents=True, exist_ok=True)
    n = len(frame)
    x = np.arange(4)
    keys = ['DDM', 'RACE']
    name = labels(generator)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, cl in enumerate(keys):
        count = frame[f'{cl}_best_model'].str.split('_').str[-1].map(
            {'1': 'NoAtt', '2': 'AttOnly', 't': 'Mul', 'g': 'Add'}).value_counts()
        rect = ax.bar(x+(k-.5)*.36, [count.get(t, 0) for t in TYPES], .36, label=cl)
        ax.bar_label(rect)
    ax.set(xticks=x, xticklabels=TYPES, title=f'{name}: four-model winners', ylabel='Simulated participants')
    ax.legend()
    saved_figure(fig, out/'01_winner_counts')

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, cl in enumerate(keys):
        avg = [frame[f'{cl}_p_{t}'].mean() for t in TYPES]
        ax.bar(x+(k-.5)*.36, avg, .36, label=cl)
    ax.set(xticks=x, xticklabels=TYPES, ylim=(0,1),
           title=f'{name}: mean model posterior probabilities', ylabel='Mean posterior probability')
    ax.legend()
    saved_figure(fig, out/'02_mean_model_probabilities')

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cl in keys:
        z=frame[f'{cl}_logBF_Mul_Add'].to_numpy(float)
        ax.hist(z, bins=28, alpha=.55, label=f'{cl} (mean {z.mean():+.2f})')
    ax.axvline(0, ls='--', color='gray')
    ax.set(title=f'{name}: Mul/Add evidence', xlabel='log BF(Mul/Add)', ylabel='Simulated participants')
    ax.legend()
    saved_figure(fig, out/'03_logBF_distribution')

    xvals=frame['RACE_logBF_Mul_Add'].to_numpy(float)
    yvals=frame['DDM_logBF_Mul_Add'].to_numpy(float)
    fig, ax=plt.subplots(figsize=(6,6))
    ax.scatter(xvals, yvals, s=15, alpha=.45)
    low=float(min(xvals.min(),yvals.min()))
    high=float(max(xvals.max(),yvals.max()))
    ax.plot([low, high], [low, high], '--', color='gray', label='Equal evidence')
    ax.axhline(0,ls=':',color='gray');ax.axvline(0,ls=':',color='gray')
    ax.set(xlabel='RACE log BF(Mul/Add)',ylabel='DDM log BF(Mul/Add)',title=f'{name}: identical simulated datasets')
    ax.legend()
    saved_figure(fig,out/'04_paired_logBF_scatter')

    delta=yvals-xvals
    fig, ax=plt.subplots(figsize=(7.5,5))
    ax.hist(delta,bins=28)
    ax.axvline(0,ls='--',color='gray')
    ax.axvline(delta.mean(),color='tab:red',label=f'Mean {delta.mean():+.2f}')
    ax.set(title=f'{name}: paired architecture difference',xlabel='DDM minus RACE log BF(Mul/Add)',ylabel='Simulated participants')
    ax.legend()
    saved_figure(fig,out/'05_paired_delta_distribution')

    param = 'theta' if mechanism(generator)=='Mul' else 'gamma'
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(frame[f'true_{param}'],delta,s=15,alpha=.4)
    ax.axhline(0,ls='--',color='gray')
    ax.set(xlabel=f'True {param}',ylabel='DDM minus RACE log BF(Mul/Add)',
           title=f'{name}: architecture bias by generating parameter')
    saved_figure(fig,out/f'06_delta_vs_{param}')

    fig,ax=plt.subplots(figsize=(8,5))
    for k,cl in enumerate(keys):
        counts = evidence_counts(frame[f'{cl}_logBF_Mul_Add'])
        vals=list(counts.values())
        ax.bar(x+(k-.5)*.36, vals, .36, label=cl)
    ax.set(xticks=x,xticklabels=['Strong Add','Weak Add','Weak Mul','Strong Mul'],
           title=f'{name}: Mul/Add BF strength',ylabel='Simulated participants')
    ax.legend()
    saved_figure(fig,out/'07_evidence_strength')

    ix=np.argsort(frame['RACE_p_Add'].to_numpy())
    fig,axes=plt.subplots(1,2,figsize=(10,max(5,min(12,n*.017))))
    for ax,cl in zip(axes,keys):
        arr=frame[[f'{cl}_p_{t}' for t in TYPES]].to_numpy(float)[ix]
        img=ax.imshow(arr,aspect='auto',vmin=0,vmax=1)
        ax.set(xticks=np.arange(4),xticklabels=TYPES,title=f'{cl} classifier',ylabel='Simulated participant')
    fig.suptitle(f'{name}: per-subject model probabilities')
    fig.colorbar(img,ax=axes,shrink=.65,label='Posterior probability')
    saved_figure(fig,out/'08_model_probability_heatmaps')


def aggregate(frames, root):
    """Create cross-architecture report for all four generators; no re-simulation."""
    if set(frames) != set(GENERATORS):
        missing = set(GENERATORS)-set(frames)
        raise ValueError(f'Combined analysis needs all generators; missing {missing}')
    out=root/'combined'
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for generator in GENERATORS:
        frame=frames[generator]
        for cl in ('DDM','RACE'):
            summaries.append(describe(frame,cl,generator))
    s=pd.DataFrame([{k:v for k,v in z.items() if not isinstance(v,dict)} for z in summaries])
    s.to_csv(out/'cross_architecture_8_conditions.csv',index=False)
    pd.concat([frames[g] for g in GENERATORS],ignore_index=True).to_csv(out/'per_simulation_all.csv',index=False)

    for cl in ('DDM','RACE'):
        mat=np.zeros((4,4),int)
        pmat=np.zeros((4,4),float)
        for r,g in enumerate(GENERATORS):
            x=frames[g]
            for c,t in enumerate(TYPES):
                mat[r,c]=int((x[f'{cl}_best_model'].str.split('_').str[-1] == ('1','2','t','g')[c]).sum())
                pmat[r,c]=float(x[f'{cl}_p_{t}'].mean())
        denom=mat.sum(axis=1,keepdims=True)
        pct=mat/np.maximum(denom,1)*100
        pd.DataFrame(mat,index=[labels(g) for g in GENERATORS],columns=TYPES).to_csv(out/f'{cl}_winning_counts.csv')
        pd.DataFrame(pct,index=[labels(g) for g in GENERATORS],columns=TYPES).to_csv(out/f'{cl}_winning_percent.csv')
        pd.DataFrame(pmat,index=[labels(g) for g in GENERATORS],columns=TYPES).to_csv(out/f'{cl}_mean_model_probabilities.csv')
        for filename,mx,fmt,scale in [(f'{cl}_winning_matrix',pct,'.1f',100),
                                       (f'{cl}_mean_prob_matrix',pmat,'.2f',1)]:
            fig,ax=plt.subplots(figsize=(8,5.7))
            img=ax.imshow(mx,aspect='auto',vmin=0,vmax=scale)
            ax.set(xticks=np.arange(4),xticklabels=TYPES,
                   yticks=np.arange(4),yticklabels=[labels(g) for g in GENERATORS],
                   xlabel='Classified as',ylabel='Generating model',title=f'{cl}: {filename.replace("_", " ")}')
            for (i,j),v in np.ndenumerate(mx):
                ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=10)
            fig.colorbar(img,ax=ax,shrink=.8)
            saved_figure(fig,out/filename)

    x=np.arange(4)
    row_names=[labels(g) for g in GENERATORS]
    for metric,title,ylabel,filename in [
        ('mean_logBF_Mul_Add','Mul/Add model evidence','Mean log BF(Mul/Add)','01_mean_logBF_comparison'),
        ('mean_logBF_correct_vs_wrong_mechanism','Evidence for actual generating mechanism','Mean log BF(correct/wrong)','02_mean_correct_evidence'),
        ('correct_mechanism_rate','Correct Mul-vs-Add preference','Fraction correct','03_mechanism_recovery'),
        ('mechanism_named_model_win_rate','Matching Mul/Add four-model winner','Fraction selecting matching mechanism','04_mechanism_named_model_winner'),
        ('strong_correct_mechanism_rate','Strong evidence for true mechanism','Fraction BF>10','05_strong_correct_evidence'),
    ]:
        fig,ax=plt.subplots(figsize=(9,5))
        for k,cl in enumerate(('DDM','RACE')):
            vals=[next(z[metric] for z in summaries if z['generator']==g and z['classifier']==cl)
                  for g in GENERATORS]
            ax.bar(x+(k-.5)*.36,vals,.36,label=cl)
        if metric in ('mean_logBF_Mul_Add','mean_logBF_correct_vs_wrong_mechanism'):
            ax.axhline(0,ls='--',color='gray')
        else:
            ax.set_ylim(0,1)
        ax.set(xticks=x,xticklabels=row_names,title=title,ylabel=ylabel)
        ax.legend()
        saved_figure(fig,out/filename)
    fig,ax=plt.subplots(figsize=(9,5))
    means=[(frames[g]['DDM_logBF_Mul_Add']-frames[g]['RACE_logBF_Mul_Add']).mean() for g in GENERATORS]
    ax.bar(row_names,means)
    ax.axhline(0,ls='--',color='gray')
    ax.set(title='Paired cross-architecture log BF difference',ylabel='DDM minus RACE log BF(Mul/Add)')
    saved_figure(fig,out/'06_paired_architecture_bias')
    results={g:paired_summary(frames[g],g,bootstrap=2000,seed=20261103+i)
             for i,g in enumerate(GENERATORS)}
    (out/'paired_statistics.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(f'[COMBINED] {out}: 4 generators x 2 classifier families, N={len(s)} conditions',flush=True)
    return s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator',choices=[*GENERATORS,'all'],default='all')
    p.add_argument('--mode',choices=['test','formal'],default='formal')
    p.add_argument('--n',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=25)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--input',type=Path)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--aggregate-only',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    if a.n<2 or a.batch_size<1 or a.bootstrap<1 or a.seed<0:
        p.error('--n>=2, --batch-size>=1, --bootstrap>=1 and --seed>=0 required')

    from experiments.wide_prior_classifier import (apply_wide_priors,
        expected_spec,verify_or_write_spec,FAMILIES as CLASS_FAMILIES,
        OUTPUT as DEFAULT_OUTPUT)
    from _shared.common import ROOT,bf,load,model_infos
    from types import SimpleNamespace
    output=Path(a.output_root) if a.output_root else DEFAULT_OUTPUT
    input_path=Path(a.input) if a.input else ROOT/'data/food_equal/trial_eye.csv'
    destination=output/'cross_architecture'/a.mode/f'seed_{a.seed}'
    requested=list(GENERATORS if a.generator=='all' else [a.generator])

    if a.aggregate_only:
        frames={}
        for generator in GENERATORS:
            path=destination/f'generated_{generator}'/'per_simulation.csv'
            if not path.is_file():
                raise FileNotFoundError(f'Cannot aggregate; missing {path}')
            frames[generator]=pd.read_csv(path)
        aggregate(frames,destination)
        return

    # Strict provenance: only checkpoints trained with the requested wide
    # parameter priors and identical empirical input may be compared.
    apply_wide_priors()
    estimators={}; ex=None
    for fam in FAMILIES:
        names=CLASS_FAMILIES[fam]
        cfg=SimpleNamespace(models=names,input=input_path,output=output,
                            mode=a.mode,seed=a.seed)
        clf_dir=output/'classifier'/'__'.join(names)/a.mode
        verify_or_write_spec(clf_dir,expected_spec(names,a.mode),'s5')
        est,design,loaded_folder=load(cfg,True)
        metadata=json.loads((clf_dir/'metadata.json').read_text(encoding='utf8'))
        if metadata.get('model_names') != names or not np.allclose(
                metadata.get('model_prior',[]),[.25]*4):
            raise ValueError(f'Incorrect model classes or training prior: {clf_dir}')
        if ex is not None and (design.sha256!=ex.sha256 or design.max_trials!=ex.max_trials):
            raise ValueError('DDM and RACE checkpoints were trained with different designs')
        estimators[fam]=est
        ex=design
        print(f'[LOAD] {fam}: {loaded_folder / "best.keras"}',flush=True)

    frames={}
    for idx,generator in enumerate(requested):
        gdir=destination/f'generated_{generator}'
        path=gdir/'per_simulation.csv'
        if path.exists() and not a.overwrite:
            raise FileExistsError(f'Already computed: {path}; use --overwrite if intended')
        gdir.mkdir(parents=True,exist_ok=True)
        # Reset C RNG independently per generator so --generator all produces
        # the same simulations as four independent --generator jobs.
        generator_seed=a.seed+500000+1000*GENERATORS.index(generator)
        ctypes.CDLL('msvcrt' if sys.platform=='win32' else None).srand(generator_seed%(2**32))
        sim,counts=ex.simulator(bf,generator,generator_seed,model_only=False)
        param='theta' if mechanism(generator)=='Mul' else 'gamma'
        pieces=[]
        for start in range(0,a.n,a.batch_size):
            chunk=min(a.batch_size,a.n-start)
            raw=sim.sample(chunk)
            conditions={key:raw[key] for key in ('trials','summary_mask','n_trials')}
            outputs={fam:classifier_output(estimators[fam],conditions) for fam in FAMILIES}
            part=pd.DataFrame({
                'generator':generator,
                'sim_id':np.arange(start,start+chunk),
                'true_d':np.asarray(raw['d']).reshape(-1),
                'true_a':np.asarray(raw['a']).reshape(-1),
                f'true_{param}':np.asarray(raw[param]).reshape(-1),
                'n_retained_trials':np.asarray(raw['summary_mask']).sum(axis=1),
            })
            for fam in FAMILIES:
                prefix=fam.upper()
                probabilities,logev=outputs[fam]
                for j,t in enumerate(TYPES):
                    part[f'{prefix}_p_{t}']=probabilities[:,j]
                names=CLASS_FAMILIES[fam]
                part[f'{prefix}_best_model']=np.asarray(names)[probabilities.argmax(axis=1)]
                part[f'{prefix}_logBF_Mul_Add']=logev[:,2]-logev[:,3]
                for i,j in PAIR_INDICES:
                    part[f'{prefix}_logBF_{TYPES[i]}_{TYPES[j]}']=logev[:,i]-logev[:,j]
            # Paired DDM-RACE differences use exactly the same simulated data.
            part['delta_DDM_minus_RACE']=part['DDM_logBF_Mul_Add']-part['RACE_logBF_Mul_Add']
            pieces.append(part)
            print(f'[EVAL] {generator}: {start+chunk}/{a.n}',flush=True)
        frame=pd.concat(pieces,ignore_index=True)
        frame.to_csv(path,index=False)
        pairwise_table(frame,generator).to_csv(gdir/'pairwise_summary.csv',index=False)
        result={
            'generator':generator,
            'n':len(frame),
            'parameter_priors':{name:list(map(float,bounds)) for name,bounds in zip(
                model_infos[generator]['free_pnames'],model_infos[generator]['free_pranges'])},
            'DDM':describe(frame,'DDM',generator),
            'RACE':describe(frame,'RACE',generator),
            'paired':paired_summary(frame,generator,a.bootstrap,a.seed+idx+700000),
            'simulation_counters':counts,
            'input_sha256':ex.sha256,
            'checkpoint_root':str(output),
            'seed':a.seed,
        }
        (gdir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        individual_plots(frame,generator,gdir)
        frames[generator]=frame
        print(f'[RESULT] {generator}: RACE BF={result["RACE"]["mean_logBF_Mul_Add"]:+.3f} '
              f'DDM BF={result["DDM"]["mean_logBF_Mul_Add"]:+.3f}',flush=True)
    if a.generator=='all':
        aggregate(frames,destination)
    print('[OUTPUT]',destination,flush=True)


if __name__=='__main__':
    main()