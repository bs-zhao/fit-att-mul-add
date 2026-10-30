from _shared.common import *
from _shared.diagnostics import parameter_summary
from _shared.data import SCALE, funcs

def behavior(rows):
    x=np.asarray(rows)*SCALE
    return dict(p_left=x[:,3].mean(),rt_mean=x[:,4].mean(),rt_q10=np.quantile(x[:,4],.1),
                rt_q50=np.quantile(x[:,4],.5),rt_q90=np.quantile(x[:,4],.9),
                gaze_left=x[:,8].mean(),fixations=x[:,9].mean())

def main():
    p=parser('Decode real participants and run posterior predictive checks',model=True)
    p.add_argument('--draws',type=positive,default=1000)
    p.add_argument('--batch-size',type=positive,default=25)
    p.add_argument('--ppc-draws',type=nonnegative,default=100)
    a=p.parse_args(); est,ex,folder=load(a)
    ids,raw,audit=ex.real(); dest=folder/'real_data'; dest.mkdir(exist_ok=True)
    pd.DataFrame(audit).to_csv(dest/'trial_audit.csv',index=False)
    names=model_infos[a.model]['free_pnames']; samples=posterior(est,raw,names,a.draws,a.batch_size)
    np.savez_compressed(dest/'posterior_samples.npz',samples=samples,subject_ids=np.asarray(ids,dtype=str),names=names)
    parameter_summary(samples,ids,names).to_csv(dest/'parameter_estimates.csv',index=False)
    if a.ppc_draws == 0:
        return
    rng=np.random.default_rng(a.seed+300000); rows=[]; details=[]
    stats=dict(trials=0,nonhits=0,maxT_excluded=0)
    for i,sid in enumerate(ids):
        observed=behavior(raw['trials'][i,raw['summary_mask'][i]])
        group=next(g for s,g in ex.groups if s==sid)
        draws=[]
        for draw in range(a.ppc_draws):
            params=funcs.make_params(model_infos[a.model],rng)
            params.update(zip(names,samples[i,int(rng.integers(a.draws))]))
            _,simrows=ex.simulate(a.model,rng,stats,params,group)
            if len(simrows)>5:
                b=behavior(simrows); draws.append(b); details.append(dict(subj=sid,draw=draw,**b))
        for metric,value in observed.items():
            vals=np.array([d[metric] for d in draws])
            rows.append(dict(subj=sid,metric=metric,observed=value,valid_ppc_draws=len(vals),
                 predictive_mean=vals.mean() if len(vals) else np.nan,
                 predictive_q05=np.quantile(vals,.05) if len(vals) else np.nan,
                 predictive_q95=np.quantile(vals,.95) if len(vals) else np.nan,
                 predictive_tail_probability=np.mean(vals>=value) if len(vals) else np.nan))
    pd.DataFrame(rows).to_csv(dest/'ppc_summary.csv',index=False)
    pd.DataFrame(details).to_csv(dest/'ppc_draws.csv',index=False)
    dump(dest/'ppc_simulation_stats.json',stats)
    if rows:
        import matplotlib.pyplot as plt
        frame=pd.DataFrame(rows); metrics=frame.metric.unique()
        fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),4),squeeze=False)
        for ax,metric in zip(axes[0],metrics):
            g=frame[frame.metric==metric]
            ax.scatter(g.observed,g.predictive_mean)
            limits=[min(g.observed.min(),g.predictive_mean.min()),max(g.observed.max(),g.predictive_mean.max())]
            ax.plot(limits,limits,color='gray'); ax.set(title=metric,xlabel='Observed',ylabel='Posterior predictive mean')
        fig.tight_layout(); fig.savefig(dest/'ppc.png',dpi=160); plt.close(fig)
if __name__=='__main__': main()
