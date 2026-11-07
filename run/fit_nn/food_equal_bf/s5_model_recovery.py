from _shared.common import *
from _shared.diagnostics import classifier_metrics, history_plot


def main():
    p = parser('Balanced held-out model recovery and probability calibration',
               classifier=True)
    p.add_argument('--n-per-model', type=positive)
    p.add_argument('--batch-size', type=positive, default=25)
    a = p.parse_args()
    try:
        names = selected_models(a)
    except ValueError as err:
        p.error(str(err))
    est, ex, folder = load(a, True)
    dest = folder / 'diagnostics'
    dest.mkdir(exist_ok=True)
    probs, truth, stats = [], [], {}
    n = a.n_per_model or (32 if a.mode == 'test' else 1000)
    for j, m in enumerate(names):
        sim, stats[m] = ex.simulator(bf, m, a.seed + 400000 + 1000*j, True)
        for start in range(0, n, a.batch_size):
            raw = sim.sample(min(a.batch_size, n-start))
            q = np.asarray(est.estimate(conditions=raw)['model_probs'])
            if q.shape != (len(raw['trials']), len(names)):
                raise RuntimeError(f'Classifier output shape mismatch: {q.shape}')
            probs.append(q)
            truth.extend([j]*len(q))
    classifier_metrics(np.concatenate(probs), np.array(truth), names, dest)
    history_plot(folder, dest)
    dump(dest / 'simulation_stats.json', stats)


if __name__ == '__main__':
    main()
