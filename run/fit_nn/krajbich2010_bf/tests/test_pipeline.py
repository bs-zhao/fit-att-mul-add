import csv
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _shared.common import *
from _shared.diagnostics import classifier_metrics
from _shared.data import feature, pack, funcs, SCALE

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        environment(42)
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'trial_eye.csv'
        with cls.path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['subj','v0','v1','response','arr_ml3_left','arr_ml3_time'])
            w.writeheader()
            for sid,n in [(1,8),(2,12)]:
                for i in range(n):
                    w.writerow(dict(subj=sid,v0=2+i%8,v1=9-i%8,response=i%2,
                                    arr_ml3_left='[0, 1, 0, 1]',arr_ml3_time='[250, 350, 300, 400]'))
        cls.ex=Experiment(cls.path)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_classifier_training_defaults_are_scoped(self):
        # Keep these defaults independent between classifier and parameter fitting.
        import inspect
        source = inspect.getsource(train)
        self.assertIn('(256 if classifier else 64)', source)
        self.assertIn('lr_patience = 4 if classifier else 8', source)
        self.assertIn('early_patience = 10 if classifier else 20', source)
        self.assertIn("batch_size=args.batch_size or", source)

    def test_legacy_feature_parity(self):
        pos,dur=[0,1,0],[100,200,300]
        expected=np.r_[[3,8],[0,1],.6,funcs.get_summary(funcs.arr2tp(pos,dur),.001)]
        np.testing.assert_allclose(feature([3,8],1,.6,pos,dur)*SCALE,expected,rtol=1e-6)
        self.assertIsNone(feature([3,8],1,29,[0,1],[15000,14000]))
        with self.assertRaises(ValueError): feature([3,8],2,.6,pos,dur)

    def test_mask_and_observed_rt(self):
        ids,raw,audit=self.ex.real()
        np.testing.assert_array_equal(raw['summary_mask'].sum(1),[8,12])
        np.testing.assert_allclose(raw['trials'][0,:8,4]*SCALE[4],1.3)
        self.assertTrue(np.all(raw['trials'][0,8:]==0))

    def test_all_eight_simulators_and_bounded_adapters(self):
        for i,model in enumerate(MODEL_NAMES):
            with self.subTest(model=model):
                sim,stats=self.ex.simulator(bf,model,100+i)
                raw=sim.sample(2); names=model_infos[model]['free_pnames']
                adapted=adapter(model)(raw)
                self.assertEqual(adapted['inference_variables'].shape,(2,len(names)))
                self.assertEqual(adapted['summary_mask'].shape,(2,12))
                self.assertTrue(np.isfinite(keras.ops.convert_to_numpy(adapted['inference_variables'])).all())
                # No mutation of empirical durations by final-fixation extension.
                self.assertEqual(self.ex.df.iloc[0].arr_du[-1],400)
                for name,(lo,hi) in zip(names,model_infos[model]['free_pranges']):
                    self.assertTrue(((raw[name]>lo)&(raw[name]<hi)).all())

    def test_train_save_reload_all_eight_and_classifier(self):
        for i,model in enumerate(MODEL_NAMES):
            with self.subTest(model=model):
                sim,_=self.ex.simulator(bf,model,800+i)
                w=workflow(sim,model)
                raw=sim.sample(4)
                w.fit_online(epochs=1,num_batches_per_epoch=1,batch_size=4,validation_data=raw,verbose=0)
                path=Path(self.temp.name)/(model+'.keras')
                w.estimator.save(path)
                restored=keras.saving.load_model(path)
                names=model_infos[model]['free_pnames']
                samples=posterior(restored,raw,names,10,2)
                self.assertTrue(np.isfinite(samples).all())
                for j,(lo,hi) in enumerate(model_infos[model]['free_pranges']):
                    self.assertTrue(((samples[:,:,j]>=lo)&(samples[:,:,j]<=hi)).all())
                keras.backend.clear_session()
        sims=[self.ex.simulator(bf,m,1800+i,True)[0] for i,m in enumerate(MODEL_NAMES)]
        mc=bf.simulators.ModelComparisonSimulator(sims)
        w=workflow(mc)
        w.fit_online(epochs=1,num_batches_per_epoch=1,batch_size=8,validation_data=mc.sample(8),verbose=0)
        path=Path(self.temp.name)/'classifier.keras'; w.estimator.save(path)
        restored=keras.saving.load_model(path)
        raw=sims[0].sample(2)
        result=restored.estimate(conditions=raw,model_prior=np.full(8,1/8))
        probs=result['model_probs']
        self.assertEqual(probs.shape,(2,8))
        np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(result['log_bayes_factors'][:,0],0,atol=1e-6)
        np.testing.assert_allclose(np.log(probs[:,1:]/probs[:,:1]),result['log_bayes_factors'][:,1:],atol=1e-5)

    def test_classifier_subset_selection_and_paths(self):
        p = parser('subset selection', classifier=True)
        a = p.parse_args(['--models', 'aRACE_g', 'aDDM_t'])
        self.assertEqual(selected_models(a), ['aDDM_t', 'aRACE_g'])
        self.assertEqual(directory(a, True).parts[-3:],
                         ('classifier', 'aDDM_t__aRACE_g', 'formal'))
        b = p.parse_args([])
        self.assertEqual(selected_models(b), MODEL_NAMES)
        self.assertEqual(directory(b, True).parts[-2:], ('classifier', 'formal'))
        with self.assertRaises(ValueError):
            selected_models(p.parse_args(['--models', 'aDDM_t']))
        with self.assertRaises(ValueError):
            selected_models(p.parse_args(['--models', 'aDDM_t', 'aDDM_t']))

    def test_subset_classifier_and_family_metrics(self):
        for names in [['aDDM_g', 'aRACE_g'],
                      ['aDDM_1', 'aDDM_t', 'aDDM_g']]:
            with self.subTest(names=names):
                sims = [self.ex.simulator(bf, m, 2100+i, True)[0]
                        for i, m in enumerate(names)]
                mc = bf.simulators.ModelComparisonSimulator(sims)
                w = workflow(mc, model_names=names)
                raw = mc.sample(6)
                w.fit_online(epochs=1, num_batches_per_epoch=1,
                             batch_size=6, validation_data=raw, verbose=0)
                ckpt = Path(self.temp.name)/('subset_'+str(len(names))+'.keras')
                w.estimator.save(ckpt)
                restored = keras.saving.load_model(ckpt)
                estimates = restored.estimate(conditions=sims[0].sample(2),
                                              model_prior=np.full(len(names),1/len(names)))
                probs = estimates['model_probs']
                self.assertEqual(probs.shape, (2,len(names)))
                np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
                np.testing.assert_allclose(np.log(probs/probs[:,:1]),
                                           estimates['log_bayes_factors'], atol=1e-5)
                dest = Path(self.temp.name)/('metrics_'+str(len(names)))
                dest.mkdir()
                classifier_metrics(np.tile(np.eye(len(names)), (2,1)),
                                   np.tile(np.arange(len(names)), 2), names, dest)
                metrics = json.loads((dest/'metrics.json').read_text())
                self.assertAlmostEqual(metrics['accuracy'],1)
                self.assertEqual(pd.read_csv(dest/'family_confusion.csv',
                                             index_col=0).values.sum(),2*len(names))
                keras.backend.clear_session()

    def test_padding_invariance(self):
        _,raw,_=self.ex.real()
        net=bf.networks.SetTransformer(summary_dim=32)
        x=raw['trials'][:1]; mask=raw['summary_mask'][:1]
        first=np.asarray(keras.ops.convert_to_numpy(net(x,mask=mask,training=False)))
        corrupted=x.copy(); corrupted[~mask]=1000
        second=np.asarray(keras.ops.convert_to_numpy(net(corrupted,mask=mask,training=False)))
        np.testing.assert_allclose(first,second,atol=1e-5,rtol=1e-5)
        perm=np.random.default_rng(3).permutation(x.shape[1])
        third=keras.ops.convert_to_numpy(net(x[:,perm],mask=mask[:,perm],training=False))
        np.testing.assert_allclose(first,third,atol=1e-5,rtol=1e-5)

if __name__=='__main__': unittest.main()
