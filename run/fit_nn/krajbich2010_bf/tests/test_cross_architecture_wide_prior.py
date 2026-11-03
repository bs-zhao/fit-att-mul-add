"""No-network tests of new-prior, cross-family model comparison summaries."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.cross_architecture_wide_prior import (
    GENERATORS, TYPES, PAIR_INDICES, classifier_output, describe,
    paired_summary, evidence_counts, pairwise_table, mechanism, family_name,
)


class DummyEstimator:
    def estimate(self, conditions, model_prior=None):
        assert np.array_equal(model_prior, np.full(4, .25))
        n=len(conditions['trials'])
        logits=np.tile(np.array([-1., 0., 2., 1.]),(n,1))
        p=np.exp(logits-logits.max(axis=1,keepdims=True))
        p/=p.sum(axis=1,keepdims=True)
        return dict(model_probs=p,log_bayes_factors=logits)


class CrossFamilyTests(unittest.TestCase):
    def test_four_generators(self):
        self.assertEqual(set(GENERATORS),
            {'aDDM_t','aDDM_g','aRACE_t','aRACE_g'})
        self.assertEqual([mechanism(g) for g in GENERATORS],
                         ['Mul','Add','Mul','Add'])
        self.assertEqual([family_name(g) for g in GENERATORS],
                         ['DDM','DDM','RACE','RACE'])

    def test_evidence_consistency_and_sign(self):
        probs,logbf=classifier_output(DummyEstimator(), {'trials':np.zeros((5,100,17))})
        self.assertEqual(probs.shape,(5,4))
        np.testing.assert_allclose(logbf[:,2]-logbf[:,3],1.0)
        self.assertEqual(evidence_counts(np.array([-3.,-1.,1.,3.])),
                         {'strong_Add':1,'weak_Add':1,'weak_Mul':1,'strong_Mul':1})

    def test_matching_mechanism_not_architecture_recovery(self):
        n=4
        frame=pd.DataFrame({
            'DDM_logBF_Mul_Add':[-3.,-2.,1.,4.],
            'RACE_logBF_Mul_Add':[-4.,-1.,1.,3.],
            'DDM_best_model':['aDDM_g','aDDM_g','aDDM_t','aDDM_t'],
            'RACE_best_model':['aRACE_g','aRACE_g','aRACE_t','aRACE_t'],
        })
        for pref in ('DDM','RACE'):
            for j,t in enumerate(TYPES):
                frame[f'{pref}_p_{t}']=.25
            for i,j in PAIR_INDICES:
                if (i,j)!=(2,3):
                    frame[f'{pref}_logBF_{TYPES[i]}_{TYPES[j]}']=float(i-j)
        for g in GENERATORS:
            f=describe(frame,'RACE',g)
            self.assertEqual(f['generator_family_is_classifier_family'],
                             g.startswith('aRACE'))
            self.assertEqual(f['mechanism_named_model_win_rate'],.5)
            pair=paired_summary(frame,g,40,77)
            self.assertEqual(pair['mean_DDM_minus_RACE_logBF'],.25)
            if g.endswith('_t'):
                self.assertEqual(pair['mean_DDM_minus_RACE_correct_mechanism_logBF'],.25)
            else:
                self.assertEqual(pair['mean_DDM_minus_RACE_correct_mechanism_logBF'],-.25)
            self.assertEqual(len(pairwise_table(frame,g)),12)


if __name__=='__main__':
    unittest.main()