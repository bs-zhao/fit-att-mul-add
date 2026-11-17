"""Numerical diagnostics shared by recovery and empirical decoding."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kstest

def heatmap(x, labels, path, title):
    fig,ax=plt.subplots(figsize=(9,7))
    im=ax.imshow(x,aspect='auto')
    ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right')
    ax.set_yticks(range(len(labels)),labels)
    ax.set_title(title)
    for (i,j),v in np.ndenumerate(x):
        ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)

def history_plot(folder, dest):
    h=pd.read_csv(folder/'history.csv')
    fig,ax=plt.subplots()
    for col in ['loss','val_loss']:
        if col in h:
            ax.plot(h[col],label=col)
    ax.legend(); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    fig.tight_layout(); fig.savefig(dest/'training_loss.png',dpi=160); plt.close(fig)

def parameter_summary(samples, ids, names):
    rows=[]
    for i,sid in enumerate(ids):
        for j,name in enumerate(names):
            s=samples[i,:,j]
            q=np.quantile(s,[.025,.05,.5,.95,.975])
            rows.append(dict(subj=sid,parameter=name,mean=s.mean(),sd=s.std(),median=q[2],
                             q025=q[0],q05=q[1],q95=q[3],q975=q[4]))
    return pd.DataFrame(rows)

def recovery(samples, truth, names, bounds, dest):
    n,draws,d=samples.shape
    estimate=samples.mean(1)
    ranks=(samples<truth[:,None,:]).sum(1)
    # Randomized SBC rank-to-uniform transform; KS p-values are descriptive, per parameter.
    u=(ranks+np.random.default_rng(919).uniform(size=ranks.shape))/(draws+1)
    fig,axes=plt.subplots(2,d,figsize=(4*d,7),squeeze=False)
    rows=[]; coverage=[]
    for j,name in enumerate(names):
        lo,hi=np.quantile(samples[:,:,j],[.05,.95],axis=1)
        err=estimate[:,j]-truth[:,j]
        axes[0,j].scatter(truth[:,j],estimate[:,j],s=12,alpha=.6)
        axes[0,j].plot(bounds[j],bounds[j],color='gray')
        axes[0,j].set(xlabel='True',ylabel='Posterior mean',title=name)
        axes[1,j].hist(u[:,j],bins=10,range=(0,1))
        axes[1,j].axhline(n/10,color='gray'); axes[1,j].set_xlabel('Randomized SBC rank')
        width=bounds[j][1]-bounds[j][0]
        rows.append(dict(parameter=name,pearson=pearsonr(truth[:,j],estimate[:,j]).statistic,
              spearman=spearmanr(truth[:,j],estimate[:,j]).statistic,rmse=np.sqrt(np.mean(err**2)),
              nrmse=np.sqrt(np.mean(err**2))/width,mae=np.mean(abs(err)),bias=err.mean(),
              coverage90=np.mean((truth[:,j]>=lo)&(truth[:,j]<=hi)),width90=np.mean(hi-lo),
              contraction=1-np.var(samples[:,:,j],axis=1).mean()/(width**2/12),
              sbc_ks=kstest(u[:,j],'uniform').statistic,sbc_ks_p=kstest(u[:,j],'uniform').pvalue))
        for level in [.5,.6,.7,.8,.9,.95]:
            low,high=np.quantile(samples[:,:,j],[(1-level)/2,(1+level)/2],axis=1)
            coverage.append(dict(parameter=name,nominal=level,empirical=np.mean((truth[:,j]>=low)&(truth[:,j]<=high))))
    fig.tight_layout(); fig.savefig(dest/'recovery_sbc.png',dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(dest/'parameter_recovery_metrics.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    pd.DataFrame(ranks,columns=names).to_csv(dest/'sbc_ranks.csv',index=False)
    frame=parameter_summary(samples,range(n),names)
    frame['truth']=truth.reshape(-1)
    frame.to_csv(dest/'parameter_recovery_long.csv',index=False)
    signed=np.mean([np.corrcoef(s.T) for s in samples],axis=0)
    absolute=np.mean([abs(np.corrcoef(s.T)) for s in samples],axis=0)
    for name,mat in [('signed',signed),('absolute',absolute)]:
        pd.DataFrame(mat,index=names,columns=names).to_csv(dest/f'posterior_correlation_{name}.csv')
        heatmap(mat,names,dest/f'posterior_correlation_{name}.png',f'Posterior correlation: {name}')
    fig,ax=plt.subplots()
    c=pd.DataFrame(coverage)
    for name,g in c.groupby('parameter'):
        ax.plot(g.nominal,g.empirical,'o-',label=name)
    ax.plot([.5,1],[.5,1],color='gray'); ax.legend(); ax.set(xlabel='Nominal coverage',ylabel='Empirical coverage')
    fig.tight_layout(); fig.savefig(dest/'coverage.png',dpi=160); plt.close(fig)

def classifier_metrics(probs, truth, names, dest):
    from .common import dump
    pred=probs.argmax(1); k=len(names)
    confusion=np.zeros((k,k),int)
    np.add.at(confusion,(truth,pred),1)
    norm=confusion/confusion.sum(1,keepdims=True)
    mean=np.stack([probs[truth==j].mean(0) for j in range(k)])
    for name,x in [('confusion_counts',confusion),('confusion_normalized',norm),('mean_probability',mean)]:
        pd.DataFrame(x,index=names,columns=names).to_csv(dest/f'{name}.csv')
        heatmap(x,names,dest/f'{name}.png',name)
    frame=pd.DataFrame(probs,columns=names)
    frame.insert(0,'true_model',np.array(names)[truth]); frame.insert(1,'pred_model',np.array(names)[pred])
    frame.to_csv(dest/'predictions.csv',index=False)
    ddm=np.array([name.startswith('aDDM') for name in names])
    true_family=np.where(ddm[truth],0,1)
    family_probs=np.stack([probs[:,ddm].sum(1),probs[:,~ddm].sum(1)],axis=1)
    fam=np.zeros((2,2),int); np.add.at(fam,(true_family,family_probs.argmax(1)),1)
    pd.DataFrame(fam,index=['DDM','ACC'],columns=['DDM','ACC']).to_csv(dest/'family_confusion.csv')
    pd.DataFrame(dict(model=names,n=confusion.sum(1),recall=norm.diagonal(),
                     mean_true_probability=mean.diagonal())).to_csv(dest/'per_model_metrics.csv',index=False)
    confidence=probs.max(1); correct=pred==truth; bins=[]
    for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=a)&((confidence<b) if b<1 else (confidence<=b))
        if mask.any():
            bins.append(dict(lower=a,upper=b,n=int(mask.sum()),confidence=float(confidence[mask].mean()),accuracy=float(correct[mask].mean())))
    bins=pd.DataFrame(bins); bins.to_csv(dest/'reliability.csv',index=False)
    fig,ax=plt.subplots(); ax.plot(bins.confidence,bins.accuracy,'o-'); ax.plot([0,1],[0,1],color='gray')
    ax.set(xlabel='Predicted confidence',ylabel='Recovery accuracy',xlim=(0,1),ylim=(0,1))
    fig.tight_layout(); fig.savefig(dest/'reliability.png',dpi=160); plt.close(fig)
    # One-vs-rest reliability detects class-specific calibration hidden by top-label ECE.
    class_bins=[]
    for j,name in enumerate(names):
        for a,b in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            mask=(probs[:,j]>=a)&((probs[:,j]<b) if b<1 else (probs[:,j]<=b))
            if mask.any():
                class_bins.append(dict(model=name,lower=a,upper=b,n=int(mask.sum()),
                                       confidence=probs[mask,j].mean(),frequency=(truth[mask]==j).mean()))
    pd.DataFrame(class_bins).to_csv(dest/'classwise_reliability.csv',index=False)
    dump(dest/'metrics.json',dict(accuracy=correct.mean(),family_accuracy=(family_probs.argmax(1)==true_family).mean(),
         log_loss=-np.log(np.clip(probs[np.arange(len(truth)),truth],1e-12,1)).mean(),
         brier=np.sum((probs-np.eye(k)[truth])**2,axis=1).mean(),
         ece=np.sum(bins.n*abs(bins.confidence-bins.accuracy))/len(truth)))
