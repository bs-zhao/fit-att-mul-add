from common import *
from diagnostics import recovery, history_plot

def main():
    p=parser('Fresh held-out parameter recovery and SBC',model=True)
    p.add_argument('--n-test',type=positive)
    p.add_argument('--draws',type=positive,default=1000)
    p.add_argument('--batch-size',type=positive,default=25)
    a=p.parse_args(); est,ex,folder=load(a)
    dest=folder/'diagnostics'; dest.mkdir(exist_ok=True)
    names=model_infos[a.model]['free_pnames']
    sim,stats=ex.simulator(bf,a.model,a.seed+200000)
    raw=sim.sample(a.n_test or (32 if a.mode=='test' else 1000))
    truth=np.concatenate([raw[n] for n in names],axis=-1)
    samples=posterior(est,raw,names,a.draws,a.batch_size)
    np.savez_compressed(dest/'recovery.npz',truth=truth,samples=samples,names=names)
    recovery(samples,truth,names,model_infos[a.model]['free_pranges'],dest)
    history_plot(folder,dest); dump(dest/'simulation_stats.json',stats)
if __name__=='__main__': main()
