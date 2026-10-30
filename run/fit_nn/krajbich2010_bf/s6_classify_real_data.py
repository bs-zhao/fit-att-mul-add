from common import *
from diagnostics import heatmap

def main():
    p=parser('Real-data model probabilities and pairwise Bayes factors')
    p.add_argument('--model-prior',type=float,nargs=8,default=[1/8]*8)
    a=p.parse_args(); est,ex,folder=load(a,True)
    prior=np.array(a.model_prior,dtype=float)
    if np.any(prior<=0) or not np.isclose(prior.sum(),1):
        p.error('--model-prior must contain eight positive probabilities summing to one')
    ids,raw,audit=ex.real(); dest=folder/'real_data'; dest.mkdir(exist_ok=True)
    pd.DataFrame(audit).to_csv(dest/'trial_audit.csv',index=False)
    # Use BayesFlow's stable log odds. Uniform training prior => log odds = log BF vs M0.
    estimates=est.estimate(conditions=raw,model_prior=np.full(8,1/8))
    logbf=np.asarray(estimates['log_bayes_factors'])
    logits=logbf+np.log(prior); logits-=logits.max(1,keepdims=True)
    probs=np.exp(logits); probs/=probs.sum(1,keepdims=True)
    df=pd.DataFrame(probs,columns=MODEL_NAMES); df.insert(0,'subj',ids)
    df['best_model']=np.array(MODEL_NAMES)[probs.argmax(1)]
    df['entropy']=-np.sum(probs*np.log(np.clip(probs,1e-12,1)),axis=1)
    df['p_DDM']=probs[:,:4].sum(1); df['p_ACC']=probs[:,4:].sum(1)
    df.to_csv(dest/'model_probabilities.csv',index=False)
    rows=[]
    for s,sid in enumerate(ids):
        for i,mi in enumerate(MODEL_NAMES):
            for j,mj in enumerate(MODEL_NAMES):
                if i!=j: rows.append(dict(subj=sid,numerator=mi,denominator=mj,log_bayes_factor=float(logbf[s,i]-logbf[s,j])))
    pd.DataFrame(rows).to_csv(dest/'pairwise_log_bayes_factors.csv',index=False)
    dump(dest/'model_prior.json',dict(training_prior=[1/8]*8,inference_prior=prior.tolist()))
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,max(4,len(ids)*.25)))
    im=ax.imshow(probs,aspect='auto',vmin=0,vmax=1)
    ax.set_xticks(range(8),MODEL_NAMES,rotation=45,ha='right'); ax.set_yticks(range(len(ids)),ids)
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(dest/'model_probabilities.png',dpi=160); plt.close(fig)
if __name__=='__main__': main()
