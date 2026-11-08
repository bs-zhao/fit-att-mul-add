from _shared.common import *


def main():
    p = parser('Real-data model probabilities and pairwise Bayes factors',
               classifier=True)
    p.add_argument('--model-prior', type=float, nargs='+',
                   help='K probabilities in canonical selected-model order; '
                        'default uniform')
    a = p.parse_args()
    try:
        names = selected_models(a)
    except ValueError as err:
        p.error(str(err))
    k = len(names)
    prior = np.asarray(a.model_prior if a.model_prior is not None else [1/k]*k,
                       dtype=float)
    if (prior.shape != (k,) or not np.isfinite(prior).all()
            or np.any(prior <= 0) or not np.isclose(prior.sum(), 1)):
        p.error(f'--model-prior requires exactly {k} finite, positive '
                'probabilities summing to one, in canonical model order')
    est, ex, folder = load(a, True)
    ids, raw, audit = ex.real()
    dest = folder / 'real_data'
    dest.mkdir(exist_ok=True)
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(dest / 'trial_audit.csv', index=False)
    # Row-level provenance for excluded real trials; source data are untouched.
    exclusion_columns = (
        'subj', 'row_index', 'reason', 'raw_choice', 'raw_rt_s',
        'n_fixations', 'total_fixation_ms'
    )
    excluded = pd.DataFrame(ex.real_exclusions, columns=exclusion_columns)
    excluded.to_csv(dest / 'excluded_real_trials.csv', index=False)
    reason_counts = excluded['reason'].value_counts().to_dict()
    print('[REAL TRIAL AUDIT] input=', int(audit_frame['input_trials'].sum()),
          'retained=', int(audit_frame['retained_trials'].sum()),
          'excluded=', len(excluded),
          'participants_included=', len(ids),
          'reasons=', reason_counts, flush=True)
    # Uniform classifier training prior; reference category is names[0].
    estimates = est.estimate(conditions=raw, model_prior=np.full(k, 1/k))
    logbf = np.asarray(estimates['log_bayes_factors'])
    if logbf.shape != (len(ids), k):
        raise RuntimeError(f'Classifier log-BF shape mismatch: {logbf.shape}')
    logits = logbf + np.log(prior)
    logits -= logits.max(1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    df = pd.DataFrame(probs, columns=names)
    df.insert(0, 'subj', ids)
    df['best_model'] = np.asarray(names)[probs.argmax(1)]
    df['entropy'] = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)), axis=1)
    ddm = np.array([m.startswith('aDDM') for m in names])
    df['p_DDM'] = probs[:, ddm].sum(1)
    df['p_ACC'] = probs[:, ~ddm].sum(1)
    df.to_csv(dest / 'model_probabilities.csv', index=False)
    rows = []
    for s, sid in enumerate(ids):
        for i, mi in enumerate(names):
            for j, mj in enumerate(names):
                if i != j:
                    rows.append(dict(subj=sid, numerator=mi, denominator=mj,
                                     log_bayes_factor=float(logbf[s,i]-logbf[s,j])))
    pd.DataFrame(rows).to_csv(dest / 'pairwise_log_bayes_factors.csv',
                              index=False)
    dump(dest / 'model_prior.json',
         dict(model_names=names, training_prior=[1/k]*k,
              inference_prior=prior.tolist()))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(7, 1.2*k), max(4, len(ids)*.25)))
    im = ax.imshow(probs, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(k), names, rotation=45, ha='right')
    ax.set_yticks(range(len(ids)), ids)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(dest / 'model_probabilities.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
