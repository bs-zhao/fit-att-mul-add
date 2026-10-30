"""Sequential end-to-end driver; each stage can also run independently."""
import argparse
import subprocess
import sys
from pathlib import Path
from data import MODEL_NAMES

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['test','formal'],default='formal')
    p.add_argument('--input'); p.add_argument('--output')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(); here=Path(__file__).resolve().parent
    base=['--mode',a.mode]
    for name in ['input','output']:
        if getattr(a,name): base+=['--'+name,str(Path(getattr(a,name)).resolve())]
    def run(script,*extra):
        subprocess.run([sys.executable,str(here/script),*base,*extra],check=True)
    for model in MODEL_NAMES:
        run('s1_train_parameter_posterior.py','--model',model,*(['--overwrite'] if a.overwrite else []))
        run('s2_parameter_recovery.py','--model',model,*(['--draws','100'] if a.mode=='test' else []))
        run('s3_decode_real_data.py','--model',model,*(['--draws','100','--ppc-draws','5'] if a.mode=='test' else []))
    run('s4_train_model_classifier.py',*(['--overwrite'] if a.overwrite else []))
    run('s5_model_recovery.py'); run('s6_classify_real_data.py')
