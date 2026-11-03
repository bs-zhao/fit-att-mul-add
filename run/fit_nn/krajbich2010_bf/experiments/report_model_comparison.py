"""Produce complete current-prior four-model comparison reports without retraining.

Reads the S5 diagnostics and S6 real_data CSV files for the given checkpoints.
Saves model comparison, subject-wise evidence, recovery and calibration figures
and CSV/JSON summaries. Never mixes prior versions or labels graphics 'old'.

Standalone: python -u experiments/report_model_comparison.py --family both
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / 'outputs/krajbich2010_bf/prior_sensitivity/gamma_0_5_theta_0_1'
MODEL_TYPES = ['NoAtt', 'AttOnly', 'Mul', 'Add']
FAMILIES = {
    'race': ['aRACE_1', 'aRACE_2', 'aRACE_t', 'aRACE_g'],
    'ddm': ['aDDM_1', 'aDDM_2', 'aDDM_t', 'aDDM_g'],
}
THRESHOLD = float(np.log(10.0))


def family_folder(output_root, mode, family):
    return Path(output_root) / 'classifier' / '__'.join(FAMILIES[family]) / mode


def plot_save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def heatmap(matrix, xticks, yticks, title, path, textfmt='.2f', vmin=None, vmax=None,
            xlabel=None, ylabel=None, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticks)), xticks, rotation=25, ha='right')
    ax.set_yticks(range(len(yticks)), yticks)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    for (i, j), x in np.ndenumerate(matrix):
        ax.text(j, i, format(x, textfmt), ha='center', va='center', fontsize=10,
                color='white' if im.norm(x) > .65 else 'black')
    fig.colorbar(im, ax=ax, shrink=.8)
    plot_save(fig, path)


def model_label(fullname):
    ending = fullname.rsplit('_', 1)[-1]
    return {'1': 'NoAtt', '2': 'AttOnly', 't': 'Mul', 'g': 'Add'}[ending]


def load_family(output_root, mode, family):
    folder = family_folder(output_root, mode, family)
    paths = [folder / 'real_data/model_probabilities.csv',
             folder / 'real_data/pairwise_log_bayes_factors.csv',
             folder / 'diagnostics/predictions.csv',
             folder / 'diagnostics/metrics.json',
             folder / 'diagnostics/reliability.csv',
             folder / 'diagnostics/classwise_reliability.csv']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Run S5 and S6 for '+family+' first; missing:\n'+'\n'.join(missing))
    names = FAMILIES[family]
    p = pd.read_csv(paths[0])
    pairwise = pd.read_csv(paths[1])
    predictions = pd.read_csv(paths[2])
    metrics = json.loads(paths[3].read_text(encoding='utf8'))
    reliability = pd.read_csv(paths[4])
    classwise = pd.read_csv(paths[5])
    if p.subj.isna().any() or p.subj.duplicated().any():
        raise ValueError(f'{family} model probabilities have null/duplicate subject IDs')
    for name in names:
        if name not in p or name not in predictions:
            raise ValueError(f'Missing {name} in model probabilities or S5 predictions')
    q = p[names].to_numpy(dtype=float)
    if not np.isfinite(q).all() or not np.allclose(q.sum(1), 1.0, atol=1e-4):
        raise ValueError(f'{family} real-data posterior model probabilities are invalid')
    if not np.array_equal(np.asarray(names)[q.argmax(1)], p.best_model.to_numpy()):
        raise ValueError(f'{family} best_model disagrees with probabilities')
    target = pairwise[(pairwise.numerator == names[2]) &
                      (pairwise.denominator == names[3])]
    if len(target) != len(p) or target.subj.duplicated().any():
        raise ValueError('Missing or duplicated Mul/Add pairwise Bayes factors')
    evidence = p[['subj']].merge(target[['subj', 'log_bayes_factor']],
                                 on='subj', validate='one_to_one', how='left')
    if evidence.log_bayes_factor.isna().any():
        raise ValueError('Missing subject-specific Mul/Add log BF')
    p = p.copy()
    p['logBF_Mul_Add'] = evidence.log_bayes_factor.to_numpy(dtype=float)
    if not np.allclose(p['logBF_Mul_Add'], np.log(np.maximum(q[:,2],1e-300))-
                       np.log(np.maximum(q[:,3],1e-300)), atol=2e-2, rtol=2e-3):
        # For extreme posteriors, the explicit S6 log BFs remain authoritative.
        if np.any((q[:,2]>1e-6)&(q[:,3]>1e-6)):
            sel = (q[:,2]>1e-6)&(q[:,3]>1e-6)
            if not np.allclose(p['logBF_Mul_Add'].to_numpy()[sel],
                               np.log(q[sel,2])-np.log(q[sel,3]), atol=2e-2):
                raise ValueError('Reported Bayes factors disagree with posterior ratios')
    if not predictions.true_model.isin(names).all() or not predictions.pred_model.isin(names).all():
        raise ValueError(f'{family} predictions contain nonmatching model names')
    y = predictions.true_model.map({n:i for i,n in enumerate(names)}).to_numpy(dtype=int)
    yhat = predictions.pred_model.map({n:i for i,n in enumerate(names)}).to_numpy(dtype=int)
    predq = predictions[names].to_numpy(dtype=float)
    if not np.isfinite(predq).all() or not np.allclose(predq.sum(1), 1, atol=1e-4):
        raise ValueError('Invalid S5 posterior probabilities')
    confusion = np.zeros((4,4), int)
    np.add.at(confusion, (y, yhat), 1)
    normal = confusion/np.maximum(confusion.sum(1, keepdims=True), 1)
    return dict(family=family, folder=folder, p=p, pairwise=pairwise,
                diag=metrics, reliability=reliability, classwise=classwise,
                predictions=predictions, q=q, y=y, predq=predq,
                confusion=confusion, normal=normal)


def evidence_groups(z):
    z = np.asarray(z, dtype=float)
    return np.array([(z < -THRESHOLD).sum(),
                     ((z >= -THRESHOLD)&(z < 0)).sum(),
                     ((z >= 0)&(z <= THRESHOLD)).sum(),
                     (z > THRESHOLD).sum()], dtype=int)


def pairwise_matrix(frame, models):
    mean = np.zeros((4,4), dtype=float)
    pct = np.full((4,4), 50., dtype=float)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                continue
            subset = frame[(frame.numerator == mi) & (frame.denominator == mj)]
            if subset.empty:
                raise ValueError(f'Pairwise comparison missing {mi}/{mj}')
            values = subset.log_bayes_factor.to_numpy(dtype=float)
            mean[i,j] = values.mean()
            pct[i,j] = 100*(values>0).mean()
    return mean,pct


def family_report(d, out):
    out.mkdir(parents=True, exist_ok=True)
    family = d['family']; names = FAMILIES[family]
    P = d['p']; q = d['q']; z = P.logBF_Mul_Add.to_numpy(dtype=float)
    N = len(P); x = np.arange(4)
    win = np.array([(P.best_model == name).sum() for name in names])
    pp = q.mean(0); se = q.std(0,ddof=1)/np.sqrt(N) if N > 1 else np.zeros(4)
    fig, ax = plt.subplots(figsize=(7.5,5))
    bars=ax.bar(MODEL_TYPES,win)
    ax.bar_label(bars)
    ax.set(ylabel=f'Participants (N={N})',title=f'{family.upper()} four-model winners',ylim=(0,max(win)*1.17+1))
    plot_save(fig,out/'01_winner_counts')

    fig, ax = plt.subplots(figsize=(7.5,5))
    bars=ax.bar(MODEL_TYPES,pp,yerr=1.96*se,capsize=4)
    ax.set(ylabel='Mean posterior model probability',ylim=(0,1.05),
           title=f'{family.upper()} mean posterior probabilities (95% CI)')
    plot_save(fig,out/'02_mean_model_probability')

    order=np.argsort(-q[:,3]); mat=q[order]
    fig,ax=plt.subplots(figsize=(7, max(6.5,N*.22)))
    im=ax.imshow(mat,aspect='auto',vmin=0,vmax=1)
    ax.set(xticks=x,xticklabels=MODEL_TYPES, ylabel='Participant (ranked by P(Add))',
           title=f'{family.upper()} participant-level model probabilities')
    ax.set_yticks(range(N),P.subj.to_numpy()[order].astype(str),fontsize=7)
    fig.colorbar(im,ax=ax,label='Posterior probability')
    plot_save(fig,out/'03_participant_probability_heatmap')

    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.hist(z,bins=min(20,max(7,int(np.sqrt(N)*2))))
    ax.axvline(0,ls='--',color='black')
    ax.axvline(z.mean(),color='tab:red',label=f'Mean={z.mean():+.2f}')
    ax.set(xlabel='log BF(Mul/Add)',ylabel='Participants',title=f'{family.upper()} Mul/Add evidence distribution')
    ax.legend()
    plot_save(fig,out/'04_logBF_Mul_Add_distribution')

    cnt=evidence_groups(z); labels=['Strong Add','Weak Add','Weak Mul','Strong Mul']
    fig,ax=plt.subplots(figsize=(8,5))
    bars=ax.bar(labels,cnt)
    ax.bar_label(bars)
    ax.set(ylabel=f'Participants (N={N})',title='Mul/Add evidence strength (BF > 10 for strong)',
           ylim=(0,max(cnt)*1.2+1))
    plot_save(fig,out/'05_evidence_strength')

    fig,ax=plt.subplots(figsize=(max(10,N*.32),5.2))
    ax.bar(np.arange(N),z)
    ax.axhline(0,ls='--',color='black')
    ax.set(xticks=np.arange(N),xticklabels=P.subj.astype(str),
           xlabel='Participant',ylabel='log BF(Mul/Add)',title=f'{family.upper()} subject-specific Mul/Add evidence')
    ax.tick_params(axis='x',rotation=90,labelsize=8)
    plot_save(fig,out/'06_subject_logBF')

    means, rates = pairwise_matrix(d['pairwise'],names)
    heatmap(means,MODEL_TYPES,MODEL_TYPES,f'{family.upper()} pairwise mean log BF',
            out/'07_pairwise_mean_logBF',textfmt='+.2f',
            xlabel='Denominator',ylabel='Numerator',cmap='coolwarm')
    heatmap(rates,MODEL_TYPES,MODEL_TYPES,f'{family.upper()} pairwise preference (%)',
            out/'08_pairwise_win_rate',textfmt='.0f',vmin=0,vmax=100,
            xlabel='Denominator',ylabel='Numerator')
    pd.DataFrame(means,index=MODEL_TYPES,columns=MODEL_TYPES).to_csv(out/'pairwise_mean_logBF.csv')
    pd.DataFrame(rates,index=MODEL_TYPES,columns=MODEL_TYPES).to_csv(out/'pairwise_win_rate_pct.csv')

    for index,values,title,ylab in [
        ('09',P.entropy.to_numpy(),'Posterior entropy','Entropy (nats)'),
        ('10',q.max(1),'Maximum model posterior probability','Maximum posterior probability')]:
        fig,ax=plt.subplots(figsize=(7.5,5))
        ax.hist(values,bins=14)
        ax.set(xlabel=ylab,ylabel='Participants',title=f'{family.upper()} {title}')
        plot_save(fig,out/(index+'_'+('posterior_entropy' if index=='09' else 'max_posterior_probability')))

    heatmap(d['normal']*100,MODEL_TYPES,MODEL_TYPES,
            f'{family.upper()} S5 recovery confusion (%)',out/'11_S5_recovery_confusion',
            textfmt='.1f',vmin=0,vmax=100,
            xlabel='Recovered model',ylabel='Generating model')
    recall=np.diag(d['normal'])*100
    true_p=np.array([d['predq'][d['y']==i,i].mean()*100 for i in range(4)])
    for index,vals,title in [
        ('12',recall,'S5 recovery rate (%)'),
        ('13',true_p,'S5 mean posterior P(true model) (%)')]:
        fig,ax=plt.subplots(figsize=(7.5,5))
        bars=ax.bar(MODEL_TYPES,vals)
        ax.bar_label(bars,fmt='%.1f')
        ax.set(ylim=(0,110),ylabel='Percent',title=f'{family.upper()} {title}')
        plot_save(fig,out/(index+'_'+('S5_classwise_recovery' if index=='12' else 'S5_true_model_probability')))

    r=d['reliability']
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot([0,1],[0,1],ls='--',color='gray',label='Perfect calibration')
    ax.plot(r.confidence,r.accuracy,'o-',label=f"ECE={d['diag']['ece']:.3f}")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted confidence',ylabel='Empirical accuracy',
           title=f'{family.upper()} S5 confidence calibration')
    ax.legend()
    plot_save(fig,out/'14_S5_reliability')

    fig,ax=plt.subplots(figsize=(7.5,6))
    ax.plot([0,1],[0,1],ls='--',color='gray',label='Perfect calibration')
    for i,model in enumerate(names):
        r=d['classwise'][d['classwise'].model==model]
        ax.plot(r.confidence,r.frequency,'o-',label=MODEL_TYPES[i])
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted model probability',
           ylabel='Observed model frequency',title=f'{family.upper()} S5 per-model calibration')
    ax.legend()
    plot_save(fig,out/'15_S5_classwise_calibration')

    fig,ax=plt.subplots(figsize=(7.5,5))
    metric_names=['Accuracy','Log loss','Brier','ECE']
    metric_vals=[d['diag'][k] for k in ('accuracy','log_loss','brier','ece')]
    bars=ax.bar(metric_names,metric_vals)
    ax.bar_label(bars,fmt='%.3f')
    ax.set(title=f'{family.upper()} S5 classification metrics',ylabel='Metric value',
           ylim=(0,max(metric_vals)*1.18))
    plot_save(fig,out/'16_S5_metric_summary')

    output=dict(family=family, mode=d['folder'].name, n_real=N,
                winners={MODEL_TYPES[i]:int(win[i]) for i in range(4)},
                mean_probability={MODEL_TYPES[i]:float(pp[i]) for i in range(4)},
                mean_logBF_Mul_Add=float(z.mean()),median_logBF_Mul_Add=float(np.median(z)),
                Mul_better_than_Add=int((z>0).sum()),Add_better_than_Mul=int((z<0).sum()),
                BF_evidence_counts=dict(zip(['strong_Add','weak_Add','weak_Mul','strong_Mul'],cnt.tolist())),
                model_recovery=d['diag'])
    (out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf8')
    P[['subj','best_model','logBF_Mul_Add','entropy',*names]].to_csv(out/'per_subject_metrics.csv',index=False)
    pd.DataFrame({'model':MODEL_TYPES,'winner_count':win,'mean_probability':pp,
                  'recall_S5':recall/100,'true_model_probability_S5':true_p/100}).to_csv(
                      out/'per_model_metrics.csv',index=False)
    print(f'[REPORT] {family.upper()}: {out} (16 PNG + 16 PDF; CSV and JSON)',flush=True)
    return output


def combined_report(race,ddm,out):
    out.mkdir(parents=True, exist_ok=True)
    entries=[race,ddm]; title=['RACE','DDM']
    a=race['p'].sort_values('subj').reset_index(drop=True)
    b=ddm['p'].sort_values('subj').reset_index(drop=True)
    if len(a)!=len(b) or not np.array_equal(a.subj.to_numpy(),b.subj.to_numpy()):
        raise ValueError('RACE and DDM must contain exactly the same participants')
    n=len(a); x=np.arange(4); width=.35
    def panels(fname,values,ylabel,ttl,ylim=None):
        fig,ax=plt.subplots(figsize=(8,5))
        for i,(label,val) in enumerate(zip(title,values)):
            bars=ax.bar(x+(i-.5)*width,val,width=width,label=label)
            if all(float(z).is_integer() for z in val):
                ax.bar_label(bars)
        ax.set(xticks=x,xticklabels=MODEL_TYPES,ylabel=ylabel,title=ttl)
        if ylim: ax.set_ylim(*ylim)
        ax.legend()
        plot_save(fig,out/fname)

    panels('01_2x4_winner_counts',[
        [(d['p'].best_model==name).sum() for name in FAMILIES[d['family']]]
        for d in entries],f'Participants (N={n})','2 x 4 model comparison: winning models')
    panels('02_2x4_mean_model_probability',[
        d['q'].mean(0) for d in entries],'Mean posterior probability',
        '2 x 4 model comparison: average posterior probabilities',(0,1))
    panels('03_2x4_S5_recovery',[
        np.diag(d['normal'])*100 for d in entries],
        'Correct recovery (%)','S5 recovery by generating model',(0,110))
    panels('04_2x4_S5_true_model_probability',[
        [d['predq'][d['y']==j,j].mean()*100 for j in range(4)] for d in entries],
        'Mean P(true model) (%)','S5 recovery: confidence in true model',(0,110))

    v1=a.logBF_Mul_Add.to_numpy(dtype=float); v2=b.logBF_Mul_Add.to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(7.5,5.3))
    ax.boxplot([v1,v2],showfliers=False)
    ax.set_xticks([1,2], title)
    for i,z in enumerate([v1,v2],1):
        ax.scatter(i+np.random.default_rng(i).uniform(-.09,.09,len(z)),z,s=19,alpha=.6)
    ax.axhline(0,ls='--',color='gray')
    ax.set(ylabel='log BF(Mul/Add)',title='Mul/Add Bayes factors across participants')
    plot_save(fig,out/'05_logBF_distribution')

    fig,ax=plt.subplots(figsize=(9,5))
    bottom=np.zeros(2)
    categories=['Strong Add','Weak Add','Weak Mul','Strong Mul']
    count=np.vstack([evidence_groups(v1),evidence_groups(v2)])
    for j,cat in enumerate(categories):
        bars=ax.bar(title,count[:,j],bottom=bottom,label=cat)
        for i,bar in enumerate(bars):
            if count[i,j]: ax.text(bar.get_x()+bar.get_width()/2,bottom[i]+count[i,j]/2,
                                   str(count[i,j]),ha='center',va='center')
        bottom+=count[:,j]
    ax.set(ylabel=f'Participants (N={n})',title='Strength of Mul/Add model evidence')
    ax.legend(ncol=2,fontsize=9)
    plot_save(fig,out/'06_logBF_strength')

    fig,ax=plt.subplots(figsize=(7.2,6))
    ax.scatter(v1,v2,alpha=.75)
    ax.axhline(0,ls='--',color='gray'); ax.axvline(0,ls='--',color='gray')
    ax.set(xlabel='RACE log BF(Mul/Add)',ylabel='DDM log BF(Mul/Add)',
           title='Between-architecture agreement in attention evidence')
    plot_save(fig,out/'07_logBF_race_vs_ddm')

    arch=v2-v1
    fig,ax=plt.subplots(figsize=(7.2,5))
    ax.hist(arch,bins=15)
    ax.axvline(0,ls='--',color='black')
    ax.axvline(arch.mean(),color='tab:red',label=f'Mean = {arch.mean():+.2f}')
    ax.set(xlabel='DDM minus RACE log BF(Mul/Add)',ylabel='Participants',
           title='Cross-architecture contrast in attention evidence')
    ax.legend();plot_save(fig,out/'08_architecture_delta_logBF')

    fig,ax=plt.subplots(figsize=(max(10,n*.31),5))
    ids=a.subj.astype(str).to_numpy()
    ax.plot(np.arange(n),v1,'o-',label='RACE',markersize=4)
    ax.plot(np.arange(n),v2,'o-',label='DDM',markersize=4)
    ax.axhline(0,ls='--',color='gray')
    ax.set(xticks=np.arange(n),xticklabels=ids,
           xlabel='Participant',ylabel='log BF(Mul/Add)',
           title='Participant-specific attention mechanism evidence')
    ax.tick_params(axis='x',rotation=90,labelsize=8)
    ax.legend();plot_save(fig,out/'09_participant_logBF_both')

    fig,ax=plt.subplots(figsize=(7.2,6))
    ax.scatter(a[FAMILIES['race'][3]],b[FAMILIES['ddm'][3]],alpha=.75)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='RACE posterior P(Add)',ylabel='DDM posterior P(Add)',
           xlim=(0,1),ylim=(0,1),title='Agreement in Add posterior probability')
    plot_save(fig,out/'10_pAdd_race_vs_ddm')

    winner=np.zeros((4,4),dtype=int)
    for an,bn in zip(a.best_model,b.best_model):
        winner[FAMILIES['race'].index(an),FAMILIES['ddm'].index(bn)]+=1
    heatmap(winner,MODEL_TYPES,MODEL_TYPES,'RACE and DDM winning-model agreement',
            out/'11_winner_agreement',textfmt='d',xlabel='DDM winner',ylabel='RACE winner')

    fig,ax=plt.subplots(figsize=(7.2,6))
    for name,d in zip(title,entries):
        r=d['reliability']
        ax.plot(r.confidence,r.accuracy,'o-',label=f'{name}, ECE={d["diag"]["ece"]:.3f}')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Confidence',ylabel='Empirical recovery accuracy',
           title='S5 confidence calibration')
    ax.legend();plot_save(fig,out/'12_S5_calibration')

    fig,ax=plt.subplots(figsize=(8,5))
    keys=['accuracy','log_loss','brier','ece']; names=['Accuracy','Log loss','Brier','ECE']
    for i,d in enumerate(entries):
        ax.bar(x+(i-.5)*width,[d['diag'][k] for k in keys],width,label=title[i])
    ax.set(xticks=x,xticklabels=names,ylabel='Metric value',
           title='S5 classifier scoring metrics')
    ax.legend();plot_save(fig,out/'13_S5_scoring')

    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.boxplot([a.entropy.to_numpy(),b.entropy.to_numpy()],showfliers=False)
    ax.set_xticks([1,2], title)
    ax.set(ylabel='Posterior entropy (nats)',title='Model comparison uncertainty')
    plot_save(fig,out/'14_posterior_entropy')

    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.boxplot([race['q'].max(1),ddm['q'].max(1)],showfliers=False)
    ax.set_xticks([1,2], title)
    ax.set(ylabel='Maximum posterior probability',title='Concentration of model evidence')
    plot_save(fig,out/'15_posterior_concentration')

    joined=pd.DataFrame({'subj':a.subj, 'RACE_best':a.best_model,
                         'DDM_best':b.best_model,'RACE_logBF_Mul_Add':v1,
                         'DDM_logBF_Mul_Add':v2,
                         'DDM_minus_RACE_logBF':arch,
                         'RACE_P_Add':a[FAMILIES['race'][3]],
                         'DDM_P_Add':b[FAMILIES['ddm'][3]]})
    joined.to_csv(out/'per_subject_cross_architecture.csv',index=False)
    result=dict(n_real=n,mean_RACE_logBF=float(v1.mean()),
                mean_DDM_logBF=float(v2.mean()),
                mean_DDM_minus_RACE=float(arch.mean()),
                n_DDM_gt_RACE=int((arch>0).sum()),
                RACE_winner_counts=race['p'].best_model.value_counts().to_dict(),
                DDM_winner_counts=ddm['p'].best_model.value_counts().to_dict())
    if n>1:
        from scipy import stats
        test=stats.ttest_rel(v2,v1)
        result['paired_t']=float(test.statistic)
        result['paired_p_two_sided']=float(test.pvalue)
        result['delta_mean_ci95']=[float(v) for v in stats.t.interval(
            .95,n-1,loc=arch.mean(),scale=stats.sem(arch))]
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(f'[REPORT] RACE+DDM combined: {out} (15 PNG + 15 PDF)',flush=True)
    return result


def generate(output_root=DEFAULT_OUTPUT,mode='formal',family='both'):
    output_root=Path(output_root)
    root=output_root/'reports'/mode
    if family not in ('race','ddm','both'):
        raise ValueError('family must be race, ddm or both')
    if family == 'both':
        data={f:load_family(output_root,mode,f) for f in ('race','ddm')}
        for f,d in data.items(): family_report(d,root/f)
        return combined_report(data['race'],data['ddm'],root/'combined')
    else:
        return family_report(load_family(output_root,mode,family),root/family)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--mode',choices=['formal','test'],default='formal')
    ap.add_argument('--family',choices=['race','ddm','both'],default='both')
    a=ap.parse_args()
    generate(a.output_root,a.mode,a.family)


if __name__=='__main__':
    main()